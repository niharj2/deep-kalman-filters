import torch
import torch.nn as nn

from .tublet_embed import TubeletPatchEmbed
from .masking import random_masking
from .vit_blocks import TransformerEncoder, TransformerDecoder

class VideoMAE(nn.Module):
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
        dec_dim=192,
        dec_depth=4,
        dec_heads=6,
        mlp_ratio=4.0,
        mask_ratio=0.9,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size

        # Patch embedding
        self.patch_embed = TubeletPatchEmbed(in_chans, enc_dim, patch_size, tubelet_size)

        # Positional embeddings (learned)
        # NOTE: we allocate at runtime once we know N, for flexibility with T/H/W
        self.pos_embed_enc = None
        self.pos_embed_dec = None

        # Encoder
        self.encoder = TransformerEncoder(enc_depth, enc_dim, enc_heads, mlp_ratio)

        # Decoder setup
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim, bias=True) if enc_dim != dec_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = TransformerDecoder(dec_depth, dec_dim, dec_heads, mlp_ratio)

        # Predict pixel-space tubelet patch (per token)
        # target patch dimension P = 3 * tubelet_size * patch_size * patch_size
        self.pred_dim = in_chans * tubelet_size * patch_size * patch_size
        self.head = nn.Linear(dec_dim, self.pred_dim)

    def _init_pos_embeds_if_needed(self, N, enc_dim, dec_dim, device):
        if self.pos_embed_enc is None or self.pos_embed_enc.shape[1] != N:
            self.pos_embed_enc = nn.Parameter(torch.zeros(1, N, enc_dim, device=device))
            nn.init.normal_(self.pos_embed_enc, std=0.02)
        if self.pos_embed_dec is None or self.pos_embed_dec.shape[1] != N:
            self.pos_embed_dec = nn.Parameter(torch.zeros(1, N, dec_dim, device=device))
            nn.init.normal_(self.pos_embed_dec, std=0.02)

    def patchify_target(self, videos: torch.Tensor):
        """
        videos: [B, 3, T, H, W]
        Returns target patches in token order: [B, N, P]
        P = 3 * t * p * p
        Token order matches the Conv3D grid order.
        """
        B, C, T, H, W = videos.shape
        t, p = self.tubelet_size, self.patch_size

        Tp, Hp, Wp = T // t, H // p, W // p
        # reshape into tubelets
        x = videos.view(B, C, Tp, t, Hp, p, Wp, p)          # [B,C,Tp,t,Hp,p,Wp,p]
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()  # [B,Tp,Hp,Wp,C,t,p,p]
        x = x.view(B, Tp * Hp * Wp, C * t * p * p)          # [B, N, P]
        return x

    def forward(self, videos: torch.Tensor):
        """
        videos: [B, 3, T, H, W]
        Returns: loss, pred, mask
          pred: [B, N, P] (decoded patches)
          mask: [B, N] (1=masked)
        """
        tokens, _grid = self.patch_embed(videos)            # [B, N, enc_dim]
        B, N, enc_dim = tokens.shape

        # Init position embeddings if needed
        self._init_pos_embeds_if_needed(N, enc_dim, self.head.in_features, tokens.device)

        # Encoder positional embeddings
        tokens = tokens + self.pos_embed_enc

        # Masking
        x_vis, mask, ids_restore, _ids_keep = random_masking(tokens, self.mask_ratio)

        # Encode visible tokens
        latent = self.encoder(x_vis)                        # [B, N_vis, enc_dim]

        # Map to decoder dim
        dec_vis = self.enc_to_dec(latent)                   # [B, N_vis, dec_dim]

        # Rebuild full token sequence for decoder: insert mask tokens
        dec_dim = dec_vis.shape[-1]
        N_vis = dec_vis.shape[1]
        N_mask = N - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, dec_dim)

        # concatenate visible + mask tokens in shuffled order, then unshuffle back
        dec_tokens_shuffled = torch.cat([dec_vis, mask_tokens], dim=1)  # [B, N, dec_dim]
        dec_tokens = torch.gather(
            dec_tokens_shuffled,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, dec_dim)
        )                                                             # [B, N, dec_dim]

        # Decoder positional embeddings
        dec_tokens = dec_tokens + self.pos_embed_dec

        # Decode and predict patches
        dec_out = self.decoder(dec_tokens)                 # [B, N, dec_dim]
        pred = self.head(dec_out)                          # [B, N, P]

        # Target patches
        target = self.patchify_target(videos)              # [B, N, P]

        # Loss on masked patches only
        mask_ = mask.unsqueeze(-1)                         # [B, N, 1]
        loss = ((pred - target) ** 2 * mask_).sum() / (mask_.sum() + 1e-8)

        return loss, pred, mask
