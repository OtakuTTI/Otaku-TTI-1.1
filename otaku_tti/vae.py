from typing import Optional, Union
import torch
import torch.nn as nn
from diffusers import AutoencoderKL

class PretrainedVAE(nn.Module):
    SCALE_FACTOR = 0.18215

    def __init__(self, model_name: str='stabilityai/sd-vae-ft-mse', device: Union[str, torch.device]='cpu', dtype: torch.dtype=torch.float32):
        super().__init__()
        self.model_name = model_name
        self._device = torch.device(device)
        self._dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(model_name, torch_dtype=dtype)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False
        self.to(self._device)

    def train(self, mode: bool=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> torch.Tensor:
        device = next(self.vae.parameters()).device
        dtype = next(self.vae.parameters()).dtype
        images = images.to(device=device, dtype=dtype)
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample() * self.SCALE_FACTOR
        return latents.float()

    @torch.no_grad()
    def decode(self, latents: torch.Tensor, chunk_size: int=4) -> torch.Tensor:
        device = next(self.vae.parameters()).device
        dtype = next(self.vae.parameters()).dtype
        scaled_latents = (latents / self.SCALE_FACTOR).to(device=device, dtype=dtype)
        if scaled_latents.size(0) > chunk_size:
            chunks = []
            for i in range(0, scaled_latents.size(0), chunk_size):
                sub = scaled_latents[i:i + chunk_size]
                out = self.vae.decode(sub).sample.float()
                chunks.append(out)
            return torch.cat(chunks, dim=0).clamp(-1.0, 1.0)
        images = self.vae.decode(scaled_latents).sample
        return images.float().clamp(-1.0, 1.0)