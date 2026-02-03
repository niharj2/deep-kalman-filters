import torch

def random_masking(x: torch.Tensor, mask_ratio: float):
    """
    x: [B, N, D]
    Returns:
      x_vis: [B, N_vis, D]
      mask: [B, N] (1=masked, 0=visible) in original token order
      ids_restore: [B, N] indices to restore original order from shuffled order
      ids_keep: [B, N_vis] kept token indices (in shuffled order)
    """
    B, N, D = x.shape
    N_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)              # [B, N]
    ids_shuffle = torch.argsort(noise, dim=1)              # [B, N]
    ids_restore = torch.argsort(ids_shuffle, dim=1)        # [B, N]

    ids_keep = ids_shuffle[:, :N_keep]                     # [B, N_keep]
    x_vis = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    mask = torch.ones(B, N, device=x.device)
    mask[:, :N_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)    # unshuffle to original order

    return x_vis, mask, ids_restore, ids_keep
