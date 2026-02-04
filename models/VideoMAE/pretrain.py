import os
import torch
from torch.utils.data import DataLoader

from dataset_processing.pt_batches_dataset import PTBatchDataset
from models.VideoMAE.videomae import VideoMAE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ----
    ds = PTBatchDataset("/content/kalman_filters/batched_data")
    # Each dataset item is a whole batch tensor, so DataLoader batch_size MUST be 1.
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)

    # ---- model ----
    model = VideoMAE(
        img_size=112,
        frames=32,
        patch_size=16,
        tubelet_size=2,
        enc_dim=384,
        enc_depth=6,
        enc_heads=6,
        dec_dim=192,
        dec_depth=4,
        dec_heads=6,
        mask_ratio=0.9,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)

    model.train()
    for step, videos in enumerate(loader):
        # videos shape from loader: [1, B, 3, T, H, W] because batch_size=1 here
        videos = videos.squeeze(0).to(device)  # -> [B,3,T,H,W]

        loss, _, _ = model(videos)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 10 == 0:
            print(f"step {step:05d} | loss {loss.item():.4f}")

        # optional: stop early for smoke test
        # if step == 50: break

    CKPT_DIR = "/content/drive/MyDrive/kalman_filters/checkpoints"
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"model": model.state_dict()}, f"{CKPT_DIR}/videomae_pretrain.pt")
    print("saved checkpoints/videomae_pretrain.pt")

if __name__ == "__main__":
    main()
