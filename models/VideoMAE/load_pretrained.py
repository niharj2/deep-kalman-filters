import torch

def load_encoder_from_full_videomae_ckpt(videomae_model, ckpt_path, device="cpu"):
    """
    Loads only encoder-related weights from a full VideoMAE checkpoint.
    Expects you saved {"model": state_dict}.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model"] if "model" in ckpt else ckpt

    # Load what matters for stage-2
    missing, unexpected = videomae_model.load_state_dict(sd, strict=False)

    # We EXPECT unexpected/missing because we won't use decoder/head in stage-2
    print("Loaded with strict=False")
    print("Missing keys (ok):", [k for k in missing if not k.startswith(("decoder", "head"))][:10])
    print("Unexpected keys (ok):", unexpected[:10])
    return videomae_model
