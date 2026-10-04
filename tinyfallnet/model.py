"""TinyFallNet: 1D-CNN that replaces 'hand-crafted features + RandomForest' on the ESP32.

Input  : raw window (150 samples = 3 s, 50 Hz) around an impact candidate, acc (g) + gyro (deg/s), any mounting.
Front  : 6 mounting-invariant channels, computed relative to the PRE-impact gravity direction u (mean acc of samples 0..34):
         |a|-1, a.u-1 (vertical), |a - (a.u)u| (horizontal), cos(tilt) = a.u/|a|, |g.u|/200 (turning), |g x u|/200 (tipping over)
Body   : conv(6->16,k5,s2) -> dwsep(16->24,s2) -> dwsep(24->32,s2) -> dwsep(32->32,s2) -> flatten(32x10) -> fc16 -> fc1
         Flatten (not global pooling) keeps WHEN things happen: the window is aligned on the impact, so 'posture after vs before' is learnable.
~70k MAC, ~9k params (36 KB float32)."""
import torch, torch.nn as nn

N_CH, WIN = 6, 150


def front(x):
    """x: (B, 150, 6) raw acc+gyro -> (B, 6, 150) invariant channels.  Mirrored exactly in the C code."""
    a, g = x[..., :3], x[..., 3:]
    u = a[:, :35].mean(1, keepdim=True); u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    an = a.norm(dim=-1); av = (a * u).sum(-1); ah = (a - av.unsqueeze(-1) * u).norm(dim=-1)
    gv = (g * u).sum(-1); gh = (g - gv.unsqueeze(-1) * u).norm(dim=-1)
    return torch.stack([an - 1, av - 1, ah, av / (an + 1e-6), gv.abs() / 200, gh / 200], 1)


def conv_bn(i, o, k, s, groups=1):
    return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm1d(o), nn.ReLU())


class DWSep(nn.Sequential):
    def __init__(s, i, o, k=5, st=2):
        super().__init__(conv_bn(i, i, k, st, groups=i), conv_bn(i, o, 1, 1))


N_PHYS = 6


def physics(x):
    """6 cheap physics scalars (same quantities the RF found most useful), from the raw window x (B,150,6).
    impact at sample 50; 'late' = last 1 s (samples 100..149)."""
    a, g = x[..., :3], x[..., 3:]
    u = a[:, :35].mean(1); u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    p = a[:, 100:].mean(1); p = p / (p.norm(dim=-1, keepdim=True) + 1e-6)
    an, gn = a.norm(dim=-1), g.norm(dim=-1)
    return torch.stack([(u * p).sum(-1),                       # cos(tilt) pre vs late posture
                        an[:, 100:].std(1) * 5,                # late stillness of |a|
                        gn[:, 100:].mean(1) / 200,             # late rotation
                        gn[:, 40:100].max(1).values / 200,     # peak rotation around the impact
                        an[:, 20:50].min(1).values,                 # free-fall dip before impact
                        an[:, 0:35].std(1) * 5], 1)            # activity before the event


class TinyFallNet(nn.Module):
    def __init__(s, phys=False):
        super().__init__(); s.phys = phys
        s.body = nn.Sequential(conv_bn(N_CH, 16, 5, 2), DWSep(16, 24), DWSep(24, 32), DWSep(32, 32))      # T: 150->75->38->19->10
        s.flat = nn.Sequential(nn.Flatten(), nn.Dropout(0.3))
        s.fc1 = nn.Linear(32 * 10 + (N_PHYS if phys else 0), 16); s.fc2 = nn.Linear(16, 1)

    def forward(s, x):
        h = s.flat(s.body(front(x)))
        if s.phys: h = torch.cat([h, physics(x)], 1)
        return s.fc2(torch.relu(s.fc1(h))).squeeze(-1)


class Wrap(nn.Module):
    """Puts the same invariant front end in front of any architecture taking (B, C, T)."""
    def __init__(s, net):
        super().__init__(); s.net = net

    def forward(s, x):
        return s.net(front(x))


def count_macs(model, T=WIN):
    """multiply-accumulates of one inference (conv + linear layers only; the front end is ~3k ops)."""
    macs = []
    def hook(m, i, o):
        if isinstance(m, nn.Conv1d): macs.append(o.shape[-1] * o.shape[1] * (m.in_channels // m.groups) * m.kernel_size[0])
        elif isinstance(m, nn.Linear): macs.append(o.numel() * m.in_features)          # per token when applied over a sequence
        elif isinstance(m, nn.LSTM): macs.append(i[0].shape[1] * 4 * m.hidden_size * (m.input_size + m.hidden_size) * (2 if m.bidirectional else 1) * m.num_layers)
        elif isinstance(m, nn.MultiheadAttention): L = i[0].shape[1]; macs.append(4 * L * m.embed_dim ** 2 + 2 * L * L * m.embed_dim)
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear, nn.LSTM, nn.MultiheadAttention))]
    model.eval()
    with torch.no_grad(): model(torch.randn(1, T, 6) + torch.tensor([0, 0, 1., 0, 0, 0]))
    for h in hs: h.remove()
    return int(sum(macs))
