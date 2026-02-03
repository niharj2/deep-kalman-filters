from torch.utils.data import DataLoader
from dataset_processing.EchoNetDataset import EchoNetDataset

dataset =  EchoNetDataset(csv_file="", video_dir="",split="TRAIN", frames=32, image_size=112)

loader = DataLoader(
    dataset=dataset,
    batch_size=8,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

videos, efs = next(iter(loader))

print(videos.shape)  # [8, 3, 32, 112, 112]
print(efs.shape)     # [8]
