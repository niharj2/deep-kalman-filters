import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_processing.EchoNetDataset import EchoNetDataset
from models.VideoMAE.encoder_only import VideoMAEEncoderOnly
from models.VideoMAE.load_pretrained import load_encoder_from_full_videomae_ckpt
from models.VideoMAE.stf_net import STFNet


# --------------------------------------------------
# Model
# --------------------------------------------------
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
        reps, grid = self.encoder(video)   # reps: [B,N,384]
        if self.use_stf:
            feat = self.stf(reps, grid)    # [B,768]
        else:
            feat = reps.mean(dim=1)        # [B,384]
        return self.head(feat).squeeze(-1) # [B]


# --------------------------------------------------
# Train
# --------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- DATA (MUST MATCH PRETRAIN) ----------
    ds = EchoNetDataset(
        csv_file="/content/drive/MyDrive/kalman_filters/EchoNet-Dynamic/FileList.csv",
        video_dir="/content/drive/MyDrive/kalman_filters/EchoNet-Dynamic/Videos",
        split="train",
        frames=32,        # MUST match SSL
        image_size=112,   # MUST match SSL
    )

    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # ---------- ENCODER (MUST MATCH CHECKPOINT) ----------
    encoder = VideoMAEEncoderOnly(
        img_size=112,
        frames=32,
        patch_size=16,
        tubelet_size=2,
        enc_dim=384,
        enc_depth=6,
        enc_heads=6,
    ).to(device)

    CKPT_SSL = "/content/drive/MyDrive/kalman_filters/checkpoints/videomae_pretrain.pt"
    encoder = load_encoder_from_full_videomae_ckpt(
        encoder, CKPT_SSL, device=device
    )

    # ---------- FULL MODEL ----------
    model = LVEFRegressor(
        encoder=encoder,
        use_stf=True,
        d_model=384,
    ).to(device)

    # ---------- OPTIMIZER ----------
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-3,
    )

    loss_fn = nn.MSELoss()

    # ---------- TRAIN ----------
    model.train()
    for step, (video, lvef) in enumerate(loader):
        video = video.to(device, non_blocking=True).float()
        lvef  = lvef.to(device, non_blocking=True).float()

        pred = model(video)
        loss = loss_fn(pred, lvef)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            print(f"step {step:05d} | loss {loss.item():.4f}")

    # ---------- SAVE ----------
    OUT_DIR = "/content/drive/MyDrive/kalman_filters/checkpoints"
    torch.save(
        {"model": model.state_dict()},
        f"{OUT_DIR}/lvef_finetune.pt",
    )
    print("saved checkpoints/lvef_finetune.pt")


if __name__ == "__main__":
    main()
