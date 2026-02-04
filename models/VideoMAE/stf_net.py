# models/VideoMAE/stf_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttnPool(nn.Module):
    """Sequence [B,L,D] -> [B,D]"""
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        w = self.gate(x)            # [B,L,1]
        a = torch.softmax(w, dim=1) # [B,L,1]
        return (a * x).sum(dim=1)   # [B,D]


class ResBlock2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(dim)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + h)


class ResBlock3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(dim)
        self.conv2 = nn.Conv3d(dim, dim, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(dim)

    def forward(self, x):
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.gelu(x + h)


def sinusoidal_1d_pos_embed(length: int, dim: int, device):
    """
    [length, dim] standard sin-cos positional encoding (not learned).
    """
    pos = torch.arange(length, device=device).float().unsqueeze(1)  # [L,1]
    i = torch.arange(dim, device=device).float().unsqueeze(0)       # [1,D]
    angle_rates = 1.0 / (10000 ** (2 * (i // 2) / dim))
    angles = pos * angle_rates                                      # [L,D]
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])
    return pe  # [L,D]


class STFNet(nn.Module):
    """
    Input:
      reps: [B, N, D]
      grid: (Tp, Hp, Wp) with N=Tp*Hp*Wp
    Output:
      fused: [B, 2D]
    """
    def __init__(self, d_model: int, temporal_downsample: int = 2):
        super().__init__()
        D = d_model
        self.temporal_downsample = temporal_downsample

        # Joint pathway (3D)
        self.pos3d = None  # lazy learnable [1,D,Tp,Hp,Wp]
        self.joint_block = ResBlock3D(D)
        self.joint_pool = GatedAttnPool(D)

        # Disjoint pathway (2D + temporal)
        self.spatial_block = ResBlock2D(D)

        # large-kernel conv will be created lazily when Hp,Wp known
        self.spatial_compress = None  # Conv2d(D, D, kernel_size=(Hp,Wp), stride=(Hp,Wp))

        # sparse temporal PE: sincos -> avg-adj -> linear projection (learnable adaptation)
        self.temporal_pe_proj = nn.Linear(D, D)

        self.disjoint_pool = GatedAttnPool(D)

    def _init_pos3d(self, Tp, Hp, Wp, D, device):
        if self.pos3d is None or self.pos3d.shape != (1, D, Tp, Hp, Wp):
            self.pos3d = nn.Parameter(torch.zeros(1, D, Tp, Hp, Wp, device=device))
            nn.init.normal_(self.pos3d, std=0.02)

    def _init_spatial_compress(self, Hp, Wp, D, device):
        if self.spatial_compress is None:
            self.spatial_compress = nn.Conv2d(
                D, D, kernel_size=(Hp, Wp), stride=(Hp, Wp), bias=True
            ).to(device)

    def forward(self, reps: torch.Tensor, grid):
        B, N, D = reps.shape
        Tp, Hp, Wp = grid
        assert N == Tp * Hp * Wp, f"N={N} but Tp*Hp*Wp={Tp*Hp*Wp}"

        # reps -> [B,D,Tp,Hp,Wp]
        x3 = reps.view(B, Tp, Hp, Wp, D).permute(0, 4, 1, 2, 3).contiguous()

        # -------------------------
        # Joint space-time pathway
        # -------------------------
        self._init_pos3d(Tp, Hp, Wp, D, reps.device)
        j = x3 + self.pos3d
        j = self.joint_block(j)  # ResNet-style 3D conv
        j_seq = j.permute(0, 2, 3, 4, 1).contiguous().view(B, Tp * Hp * Wp, D)
        j_feat = self.joint_pool(j_seq)  # [B,D]

        # -------------------------
        # Disjoint pathway
        # -------------------------
        # treat each time slice as separate 2D map: [B*Tp, D, Hp, Wp]
        x2 = x3.permute(0, 2, 1, 3, 4).contiguous().view(B * Tp, D, Hp, Wp)

        x2 = self.spatial_block(x2)  # 2D ResNet block

        # large-kernel conv stride (Hp,Wp) -> [B*Tp, D, 1, 1]
        self._init_spatial_compress(Hp, Wp, D, reps.device)
        x2c = self.spatial_compress(x2)
        x2c = x2c.view(B, Tp, D)  # [B,Tp,D]

        # sparse temporal PE: encode original frame indices, then avg adjacent to downsample
        # frame indices length Tp (tubelets along time)
        pe = sinusoidal_1d_pos_embed(Tp, D, device=reps.device)  # [Tp,D]
        pe = pe.unsqueeze(0).expand(B, -1, -1)                   # [B,Tp,D]

        # downsample by averaging adjacent pairs (paper says reduce length by half)
        if self.temporal_downsample == 2 and Tp % 2 == 0:
            x2c = 0.5 * (x2c[:, 0::2, :] + x2c[:, 1::2, :])     # [B,Tp/2,D]
            pe  = 0.5 * (pe[:, 0::2, :]  + pe[:, 1::2, :])      # [B,Tp/2,D]

        pe = self.temporal_pe_proj(pe)  # learnable adaptation to "sparse learnable PE"
        x2c = x2c + pe                  # add PE
        d_feat = self.disjoint_pool(x2c)  # gated attention over time -> [B,D]

        # concat
        fused = torch.cat([j_feat, d_feat], dim=-1)  # [B,2D]
        return fused
