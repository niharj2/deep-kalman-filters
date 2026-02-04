# models/VideoMAE/prediction.py

import os
import sys
import torch
import torch.nn as nn

# ---------------------------------------------------------------------
# Make project root importable when running this file directly:
#   python models/VideoMAE/prediction.py
# (Still recommended: python -m models.VideoMAE.prediction)
# ---------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset_processing.EchoNetDataset import EchoNetDataset
from models.VideoMAE.encoder_only import VideoMAEEncoderOnly
from models.VideoMAE.load_pretrained import load_encoder_from_full_videomae_ckpt
from models.VideoMAE.stf_net import STFNet

torch.set_printoptions(sci_mode=False)

# ----------------------------
# Model (must match training)
# ----------------------------
class LVEFRegressor(nn.Module):
    def __init__(self, encoder, use_stf=True, d_model=384):
        super().__init__()
        self.encoder = encoder
        self.use_stf = use_stf
        self.stf = STFNet(d_model) if use_stf else None
        in_dim = 2 * d_model if use_stf else d_model
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1),
        )

    def forward(self, video):  # [B,3,T,H,W]
        reps, grid = self.encoder(video)     # [B,N,d], grid=(Tp,Hp,Wp)
        feat = self.stf(reps, grid) if self.use_stf else reps.mean(dim=1)
        return self.head(feat).squeeze(-1)   # [B]


# ---------------------------------------------------------------------
# Checkpoint loading (robust)
# ---------------------------------------------------------------------
def _unwrap_state_dict(ckpt_obj):
    # Supports:
    #   {"model": state_dict}
    #   {"state_dict": state_dict}
    #   state_dict directly
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise TypeError(f"Unrecognized checkpoint type: {type(ckpt_obj)}")


def filter_state_dict_for_encoder_regressor(sd: dict) -> dict:
    """
    Drop keys that belong to full VideoMAE encoder-decoder pretraining checkpoints
    or positional embeds that your current encoder class does not define.
    """
    keep = {}
    for k, v in sd.items():
        # drop full pretrain-only parts
        if k.startswith("decoder."):
            continue
        if k.startswith("enc_to_dec."):
            continue
        if k in ("mask_token", "pos_embed_enc", "pos_embed_dec"):
            continue

        # common mismatch keys for pos embeds
        if k in ("encoder.pos_embed", "encoder.pos_embed_enc", "encoder.pos_embed_dec"):
            # your current encoder-only impl typically doesn't expose these
            continue

        keep[k] = v
    return keep


def load_finetune_checkpoint(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _unwrap_state_dict(ckpt)
    sd = filter_state_dict_for_encoder_regressor(sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("Loaded finetune ckpt with strict=False")
    print("missing (count):", len(missing))
    print("unexpected (count):", len(unexpected))
    if missing:
        print("missing (first 25):", list(missing)[:25])
    if unexpected:
        print("unexpected (first 25):", list(unexpected)[:25])

    return model.to(device)


# ---------------------------------------------------------------------
# Video normalization helper (prevents insane constant outputs)
# ---------------------------------------------------------------------
def normalize_video_tensor(video: torch.Tensor) -> torch.Tensor:
    """
    video expected shape: [3,T,H,W]
    - convert to float
    - if appears to be uint8 range, scale to [0,1]
    """
    video = video.float()
    vmax = float(video.max())
    if vmax > 1.5:  # likely 0..255
        video = video / 255.0
    return video


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    # ----------------------------
    # EDIT THESE PATHS IF NEEDED
    # ----------------------------
    CSV_FILE = "/content/drive/MyDrive/kalman_filters/EchoNet-Dynamic/FileList.csv"
    VIDEO_DIR = "/content/drive/MyDrive/kalman_filters/EchoNet-Dynamic/Videos"

    CKPT_SSL = "/content/drive/MyDrive/kalman_filters/checkpoints/videomae_pretrain.pt"
    CKPT_FT  = "/content/drive/MyDrive/kalman_filters/checkpoints/lvef_finetune.pt"

    # ----------------------------
    # MUST MATCH YOUR TRAINING
    # ----------------------------
    IMG_SIZE = 112
    FRAMES   = 32
    D_MODEL  = 384
    DEPTH    = 6
    HEADS    = 6
    PATCH    = 16
    TUBELET  = 2

    # If you trained with EF normalized by /100, set this True
    EF_WAS_NORMALIZED = False

    # ----------------------------
    # Build encoder + load SSL
    # ----------------------------
    encoder = VideoMAEEncoderOnly(
        img_size=IMG_SIZE,
        frames=FRAMES,
        patch_size=PATCH,
        tubelet_size=TUBELET,
        enc_dim=D_MODEL,
        enc_depth=DEPTH,
        enc_heads=HEADS,
    ).to(device)

    encoder = load_encoder_from_full_videomae_ckpt(encoder, CKPT_SSL, device=device)

    # ----------------------------
    # Build regressor + load finetune
    # ----------------------------
    model = LVEFRegressor(encoder=encoder, use_stf=True, d_model=D_MODEL).to(device)
    model = load_finetune_checkpoint(model, CKPT_FT, device=device)
    model.eval()

    # ----------------------------
    # Dataset (prediction)
    # ----------------------------
    ds = EchoNetDataset(
        csv_file=CSV_FILE,
        video_dir=VIDEO_DIR,
        split="TEST",   # make sure this matches exactly in FileList.csv
        frames=FRAMES,
        image_size=IMG_SIZE,
    )

    print("dataset size:", len(ds))
    if len(ds) == 0:
        raise ValueError(
            "Dataset length is 0. Your split string didn't match FileList.csv.\n"
            "Fix by checking df['Split'].unique() and using that exact string (case matters)."
        )

    # ----------------------------
    # Predict a few samples
    # ----------------------------
    sample_idxs = [0, min(1, len(ds) - 1), min(5, len(ds) - 1)]
    for idx in sample_idxs:
        video, gt = ds[idx]  # video: [3,T,H,W]
        video = normalize_video_tensor(video)

        # one-time sanity check (first sample)
        if idx == sample_idxs[0]:
            print("video sanity:",
                  tuple(video.shape),
                  video.dtype,
                  "min/max:", float(video.min()), float(video.max()))

        gt_val = float(gt)

        with torch.no_grad():
            raw = model(video.unsqueeze(0).to(device)).item()

        print(f"[idx={idx}] RAW={raw:.6f} | GT EF={gt_val:.2f}")

        if EF_WAS_NORMALIZED:
            pred_val = raw * 100.0   # model trained on EF/100
        else:
            pred_val = raw           # model trained on EF directly

        print(f"[idx={idx}] GT EF={gt_val:.2f} | Pred EF={pred_val:.2f}")

    print("✅ done")


if __name__ == "__main__":
    main()
