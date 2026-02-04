import torch
import torch.nn as nn

from .tublet_embed import TubeletPatchEmbed  # NOTE: match your filename exactly
from .vit_blocks import TransformerEncoder


class VideoMAEEncoderOnly(nn.Module):
    """
    Encoder-only VideoMAE.
    Input:  videos [B, 3, T, H, W]
    Output: reps   [B, N, D] and grid (Tp, Hp, Wp)
    """
    def __init__(
        self,
        img_size=112,
        frames=32,
        patch_size=16,
        tubelet_size=2,
        in_chans=3,
        enc_dim=384,
        enc_depth=6,
        enc_heads=6,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.frames = frames
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.enc_dim = enc_dim

        self.patch_embed = TubeletPatchEmbed(
            in_chans=in_chans,
            embed_dim=enc_dim,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
        )

        self.encoder = TransformerEncoder(enc_depth, enc_dim, enc_heads, mlp_ratio)

        # learned positional embedding created lazily once we know N
        self.pos_embed_enc = None

    def _init_pos_embed_if_needed(self, N, device):
        if self.pos_embed_enc is None or self.pos_embed_enc.shape[1] != N:
            self.pos_embed_enc = nn.Parameter(torch.zeros(1, N, self.enc_dim, device=device))
            nn.init.normal_(self.pos_embed_enc, std=0.02)


    def forward(self, videos: torch.Tensor):
        # videos: [B,3,T,H,W]
        tokens, grid = self.patch_embed(videos)      # [B,N,D], grid=(Tp,Hp,Wp)
        B, N, D = tokens.shape

        self._init_pos_embed_if_needed(N, tokens.device)
        tokens = tokens + self.pos_embed_enc

        reps = self.encoder(tokens)                  # [B,N,D]
        return reps, grid
