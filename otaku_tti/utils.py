import copy
import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from PIL import Image
import torch
import torch.nn as nn

def save_image_grid(images: torch.Tensor, save_path: Union[str, Path], nrow: Optional[int]=None, padding: int=4, bg_color: Tuple[int, int, int]=(20, 20, 20)) -> Image.Image:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    b, _, h, w = images.shape
    if nrow is None:
        nrow = int(math.ceil(math.sqrt(b)))
    ncol = int(math.ceil(b / nrow))
    grid_w = nrow * w + (nrow + 1) * padding
    grid_h = ncol * h + (ncol + 1) * padding
    grid = Image.new('RGB', (grid_w, grid_h), color=bg_color)
    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        x = padding + col * (w + padding)
        y = padding + row * (h + padding)
        img_tensor = images[idx].detach().cpu().float()
        img_tensor = (img_tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
        pil_img = Image.fromarray(img_np)
        grid.paste(pil_img, (x, y))
    grid.save(save_path)
    return grid

class EMA:

    def __init__(self, model: nn.Module, decay: float=0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, ema_v in self.ema_model.state_dict().items():
            model_v = msd[k].detach()
            if torch.isnan(model_v).any() or torch.isinf(model_v).any():
                continue
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        self.ema_model.load_state_dict(state_dict)
