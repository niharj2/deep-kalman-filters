import torch
import load_video
import os
import pandas as pd

class EchoNetDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, video_dir, split,
                 frames=32, image_size=224):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Split"] == split]
        self.video_dir = video_dir
        self.frames = frames
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        video_path = os.path.join(
            self.video_dir, row["FileName"]
        )

        video = load_video(
            video_path,
            frames=self.frames,
            size=self.image_size
        )

        lvef = row["EF"]

        return video, torch.tensor(lvef, dtype=torch.float32)
