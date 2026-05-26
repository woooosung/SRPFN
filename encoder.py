import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x32 = x.float()
        rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        y = x32 * rms
        return (y.to(x.dtype)) * self.gamma

class HierarchicalStateBlock(nn.Module):
    def __init__(self, fd, od):
        super().__init__()
        self.ctx_mlp = nn.Sequential(
            nn.Linear(2 * fd, 2 * fd),
            nn.GELU(),
            nn.Linear(2 * fd, fd)
        )

        self.state_mlp = nn.Sequential(
            nn.Linear(2 * fd, 2 * od),
            nn.GELU(),
            nn.Linear(2 * od, od)
        )
        self.norm = RMSNorm(od)

    def forward(self, u, i, t):
        ctx_in = torch.cat([i, t], dim=-1)
        ctx = self.ctx_mlp(ctx_in)

        state_in = torch.cat([u, ctx], dim=-1)
        state = self.state_mlp(state_in)

        return self.norm(state)

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1, eps=1e-6):
        super().__init__()
        assert input_dim % 5 == 0
        self.fd = input_dim // 5
        fd = self.fd
        od = output_dim
        self.eps = eps

        self.state_block = HierarchicalStateBlock(fd, od)

        self.flow_mlp = nn.Sequential(
            nn.Linear(2 * od, 2 * od),
            nn.GELU(),
            nn.Linear(2 * od, od)
        )

        self.gate_mlp = nn.Sequential(
            nn.Linear(4, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.final_norm = RMSNorm(od)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        nn.init.zeros_(self.gate_mlp[-1].weight)

        nn.init.normal_(self.flow_mlp[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.flow_mlp[-1].bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _cos(self, a, b):
        return F.cosine_similarity(a.float(), b.float(), dim=-1).clamp(-1 + 1e-4, 1 - 1e-4)

    def forward(self, interaction_vectors):
        squeeze = False
        if interaction_vectors.dim() == 2:
            interaction_vectors = interaction_vectors.unsqueeze(1)
            squeeze = True

        B, T, D = interaction_vectors.shape
        fd = self.fd

        x5 = interaction_vectors.view(B, T, 5, fd)
        u, i_prev, i_curr, t_prev, t_curr = x5.unbind(dim=2)

        s_prev = self.state_block(u, i_prev, t_prev)

        s_curr = self.state_block(u, i_curr, t_curr)

        flow_in = torch.cat([s_prev, s_curr], dim=-1)
        delta = self.flow_mlp(flow_in)

        inv = torch.stack([
            self._cos(i_prev, i_curr),
            self._cos(t_prev, t_curr),
            self._cos(i_prev, t_prev),
            self._cos(i_curr, t_curr),
        ], dim=-1)

        gate = torch.sigmoid(self.gate_mlp(inv.float())).to(delta.dtype)

        h = s_curr + gate * delta

        out = self.final_norm(h)
        out = self.dropout(out)

        if squeeze:
            out = out.squeeze(1)
        return out

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, interaction_vectors):
        return self.net(interaction_vectors)