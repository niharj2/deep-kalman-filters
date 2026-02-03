import torch
import torch.nn as nn

class TubeletPatchEmbed(nn.Module):
    """
    Video -> tubelet tokens using Conv3d.

    Input:  [B, 3, T, H, W]
    Output: tokens [B, N, D], grid (T', H', W')
    """
    def __init__(self, in_chans=3, embed_dim=768, patch_size=16, tubelet_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            bias=True,
        )

    def forward(self, x: torch.Tensor):
        B, C, T, H, W = x.shape
        t, p = self.tubelet_size, self.patch_size

        if T % t != 0:
            raise ValueError(f"T={T} not divisible by tubelet_size={t}")
        if H % p != 0 or W % p != 0:
            raise ValueError(f"H,W=({H},{W}) not divisible by patch_size={p}")

        x = self.proj(x)             # [B, D, T', H', W']
        B, D, Tp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x, (Tp, Hp, Wp)
