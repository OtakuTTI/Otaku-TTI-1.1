from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler, DPMSolverMultistepScheduler
from .text_encoder import CLIPTextEncoder
from .vae import PretrainedVAE

def build_otaku_diffusion_unet(sample_size: int=32, in_channels: int=4, out_channels: int=4, cross_attention_dim: int=512, block_out_channels: Tuple[int, ...]=(128, 256, 512, 512), attention_head_dim: int=32) -> UNet2DConditionModel:
    unet = UNet2DConditionModel(sample_size=sample_size, in_channels=in_channels, out_channels=out_channels, layers_per_block=2, block_out_channels=block_out_channels, down_block_types=('DownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D'), up_block_types=('CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'UpBlock2D'), cross_attention_dim=cross_attention_dim, attention_head_dim=attention_head_dim, use_linear_projection=True)
    return unet

class OtakuTTILatentDiffusion(nn.Module):

    def __init__(self, unet: Optional[UNet2DConditionModel]=None, clip_model: str='openai/clip-vit-base-patch32', vae_model: str='stabilityai/sd-vae-ft-mse', image_size: int=256, device: Union[str, torch.device]='cuda' if torch.cuda.is_available() else 'cpu', dtype: torch.dtype=torch.bfloat16):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = image_size
        self.latent_size = image_size // 8
        if unet is None:
            self.unet = build_otaku_diffusion_unet(sample_size=self.latent_size, cross_attention_dim=512)
        else:
            self.unet = unet
        self.unet.to(device=self.device, dtype=self.dtype)
        self.text_encoder = CLIPTextEncoder(model_name=clip_model, device=self.device)
        self.vae = PretrainedVAE(model_name=vae_model, device=self.device, dtype=self.dtype)
        self.train_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', prediction_type='epsilon')
        self.eval_scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', prediction_type='epsilon', algorithm_type='dpmsolver++', use_karras_sigmas=True)

    def train_step(self, real_images: torch.Tensor, prompts: List[str], uncond_prob: float=0.1) -> torch.Tensor:
        b = real_images.size(0)
        with torch.no_grad():
            latents = self.vae.encode(real_images).to(dtype=self.dtype)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.train_scheduler.config.num_train_timesteps, (b,), device=self.device).long()
        noisy_latents = self.train_scheduler.add_noise(latents, noise, timesteps)
        with torch.no_grad():
            dropped_prompts = ['' if torch.rand(1).item() < uncond_prob else p for p in prompts]
            _, token_embeds, _ = self.text_encoder.encode(dropped_prompts)
            token_embeds = token_embeds.to(dtype=self.dtype)
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states=token_embeds).sample
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction='mean')
        return loss

    @torch.no_grad()
    def generate(self, prompts: Union[str, List[str]], negative_prompts: Optional[Union[str, List[str]]]=None, num_inference_steps: int=25, guidance_scale: float=7.5, generator: Optional[torch.Generator]=None, seed: Optional[int]=None) -> torch.Tensor:
        self.unet.eval()
        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)
        if negative_prompts is None:
            negative_prompts = [''] * batch_size
        elif isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * batch_size
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        _, cond_tokens, _ = self.text_encoder.encode(prompts)
        _, uncond_tokens, _ = self.text_encoder.encode(negative_prompts)
        cond_tokens = cond_tokens.to(device=self.device, dtype=self.dtype)
        uncond_tokens = uncond_tokens.to(device=self.device, dtype=self.dtype)
        text_embeddings = torch.cat([uncond_tokens, cond_tokens], dim=0)
        self.eval_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.eval_scheduler.timesteps
        shape = (batch_size, 4, self.latent_size, self.latent_size)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        latents = latents * self.eval_scheduler.init_noise_sigma
        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.eval_scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.eval_scheduler.step(noise_pred, t, latents).prev_sample
        images = self.vae.decode(latents)
        return images