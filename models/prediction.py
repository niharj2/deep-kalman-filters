import os
import torch
import torch.nn as nn

from dataset_processing.EchoNetDataset import EchoNetDataset
from models.VideoMAE.encoder_only import VideoMAEEncoderOnly
from models.VideoMAE.load_pretrained import load_encoder_from_full_videomae_ckpt
from models.VideoMAE.stf_net import STFNet


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


def fix_key_mismatches(sd: dict) -> dict:
    """
    Fix common key mismatches between saved checkpoints and current code.
    """
    sd = dict(sd)

    # common mismatch: old code used encoder.pos_embed, new uses encoder.pos_embed_enc
    if "encoder.pos_embed" in sd and "encoder.pos_embed_enc" not in sd:
        sd["encoder.pos_embed_enc"] = sd.pop("encoder.pos_embed")

    return sd


def load_finetune_checkpoint(model: nn.Module, ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"Checkpoint missing 'model'. Keys={list(ckpt.keys())}")

    sd = fix_key_mismatches(ckpt["model"])
    missing, unexpected = model.load_state_dict(sd, strict=False)

    print("Loaded finetune ckpt with strict=False")
    print("missing (count):", len(missing))
    print("unexpected (count):", len(unexpected))
    if len(missing) > 0:
        print("missing (first 25):", list(missing)[:25])
    if len(unexpected) > 0:
        print("unexpected (first 25):", list(unexpected)[:25])

    return model


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
    # MUST MATCH YOUR SSL PRETRAIN
    # ----------------------------
    IMG_SIZE = 112
    FRAMES   = 32
    D_MODEL  = 384
    DEPTH    = 6
    HEADS    = 6
    PATCH    = 16
    TUBELET  = 2

    # If you trained with EF normalized by /100, set this True
    EF_WAS_NORMALIZED = True

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
        split="TEST",         # change to "val" or "TRAIN" if your CSV uses uppercase
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
    for idx in [0, min(1, len(ds)-1), min(5, len(ds)-1)]:
        video, gt = ds[idx]  # video: [3,T,H,W]
        with torch.no_grad():
            pred = model(video.unsqueeze(0).to(device).float()).item()

        gt_val = float(gt)
        pred_val = float(pred)

        if EF_WAS_NORMALIZED:
            pred_val = pred_val * 100.0

        print(f"[idx={idx}] GT EF={gt_val:.2f} | Pred EF={pred_val:.2f}")

    print("✅ done")


if __name__ == "__main__":
    main()
