from pathlib import Path
import torch
from torch.utils.data import Dataset

class PTBatchDataset(Dataset):
    """
    Dataset over saved batch files: each item is one batch file.
    Good for your current setup (preprocessed_batches/train/batch_XXXXX.pt).
    """
    def __init__(self, root_dir):
        self.root = Path(root_dir)
        self.files = sorted(self.root.glob("batch_*.pt"))
        if not self.files:
            raise FileNotFoundError(f"No batch_*.pt found in {self.root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu")
        videos = data["videos"]  # [B,3,T,H,W]
        return videos
