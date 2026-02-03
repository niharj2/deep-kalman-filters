import torch
import torchvision
import torchvision.transforms.functional as F

def load_video(video_path, frames=32, size=112):
    """
    Args:
        video_path (str): path to .avi file
        frames (int): number of frames to sample
        size (int): resize frames to size x size

    Returns:
        video (Tensor): shape [C, T, H, W]
    """

    # 1. Read video
    video, _, _ = torchvision.io.read_video(
        video_path, pts_unit="sec"
    )
    # video shape: [T, H, W, C]

    T, H, W, C = video.shape

    # 2. Uniformly sample frames
    if T >= frames:
        indices = torch.linspace(0, T - 1, frames).long()
        video = video[indices]
    else:
        # if video shorter than required frames, repeat last frame
        pad = frames - T
        last = video[-1].unsqueeze(0).repeat(pad, 1, 1, 1)
        video = torch.cat([video, last], dim=0)

    # 3. Convert to float and normalize to [0,1]
    video = video.float() / 255.0

    # 4. Resize frames
    video = video.permute(0, 3, 1, 2)   # [T, C, H, W]
    video = F.resize(video, [size, size])

    # 5. Rearrange to [C, T, H, W]
    video = video.permute(1, 0, 2, 3)

    return video

