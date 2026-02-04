# dataset_processing/EchoNetDataset.py

import os
import torch
import pandas as pd

from .load_video import load_video


class EchoNetDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, video_dir, split, frames=32, image_size=224):
        self.video_dir = video_dir
        self.frames = frames
        self.image_size = image_size

        # Read + filter split
        df = pd.read_csv(csv_file)
        df = df[df["Split"] == split].copy()

        # Build a lookup of actual video filenames on disk (case-insensitive)
        video_files = {
            f.lower(): f
            for f in os.listdir(self.video_dir)
            if f.lower().endswith(".avi")
        }

        def resolve(fname):
            """
            CSV usually stores base ID without extension.
            Resolve it to an existing file in Videos/.
            """
            base = str(fname).strip()
            key = (base + ".avi").lower()
            if key in video_files:
                return os.path.join(self.video_dir, video_files[key])
            return None

        df["video_path"] = df["FileName"].apply(resolve)

        # Keep only rows whose videos actually exist
        before = len(df)
        df = df[df["video_path"].notnull()].reset_index(drop=True)
        after = len(df)
        print(f"[EchoNetDataset] split={split} kept {after}/{before} rows with existing videos")

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
      row = self.df.iloc[idx]
      video_path = row["video_path"]

      if not os.path.exists(video_path):
          raise FileNotFoundError(f"Video not found (post-filter): {video_path}")

      video = load_video(
          video_path,
          frames=self.frames,
          size=self.image_size
      )

      # Hard asserts so it never returns None silently
      if video is None:
          raise RuntimeError(f"load_video returned None for: {video_path}")

      if not isinstance(video, torch.Tensor):
          raise RuntimeError(f"load_video returned {type(video)} for: {video_path}")

      if video.numel() == 0:
          raise RuntimeError(f"Loaded empty video tensor for: {video_path}")

      lvef = float(row["EF"])
      return video, torch.tensor(lvef, dtype=torch.float32)

