import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_processing.EchoNetDataset import EchoNetDataset
from models.VideoMAE.encoder_only import VideoMAEEncoderOnly
from models.VideoMAE.load_pretrained import load_encoder_from_full_videomae_ckpt
from models.VideoMAE.stf_net import STFNet  # you implement next

class LVEFRegressor(nn.Module):
    def __init__(self, encoder, use_stf=True, d_model=768):
        super().__init__()
        self.encoder = encoder
        self.use_stf = use_stf
        self.stf = STFNet(d_model) if use_stf else None
        in_dim = (2 * d_model) if use_stf else d_model
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1)
        )

    def forward(self, video):  # video: [B,3,T,H,W]
        reps, grid = self.encoder(video)   # reps: [B,L,d], grid=(t,h,w)
        if self.use_stf:
            feat = self.stf(reps, grid)    # [B,2d]
        else:
            feat = reps.mean(dim=1)        # [B,d] baseline pooling
        pred = self.head(feat).squeeze(-1) # [B]
        return pred

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset (your labeled EchoNet CSV setup)
    ds = EchoNetDataset(
        csv_file="YOUR_CSV.csv",
        video_dir="YOUR_VIDEO_DIR",
        split="train",
        frames=16,       # paper uses 16 frames :contentReference[oaicite:4]{index=4}
        image_size=224,  # paper uses 224 :contentReference[oaicite:5]{index=5}
    )
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    # Encoder-only model (no decoder)
    encoder = VideoMAEEncoderOnly(
        img_size=224,
        frames=16,
        patch_size=16,
        tubelet_size=2,  # τ=2 in paper :contentReference[oaicite:6]{index=6}
        enc_dim=768,     # d_model=768 in paper :contentReference[oaicite:7]{index=7}
        enc_depth=12,    # paper says 12 layers :contentReference[oaicite:8]{index=8}
        enc_heads=12,
    ).to(device)

    # Load your pretrained weights (from pretrain stage)
    encoder = load_encoder_from_full_videomae_ckpt(
        encoder, "checkpoints/videomae_pretrain.pt", device=device
    )

    model = LVEFRegressor(encoder, use_stf=True, d_model=768).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for step, (video, lvef) in enumerate(loader):
        video = video.to(device, non_blocking=True)
        lvef = lvef.to(device, non_blocking=True)

        pred = model(video)
        loss = loss_fn(pred, lvef)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 20 == 0:
            print(f"step {step:05d} | loss {loss.item():.4f}")

    torch.save({"model": model.state_dict()}, "checkpoints/lvef_finetune.pt")
    print("saved checkpoints/lvef_finetune.pt")

if __name__ == "__main__":
    main()
