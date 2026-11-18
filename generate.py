import argparse
from pathlib import Path
import time
from typing import List
import torch
from diffusers import DPMSolverMultistepScheduler
from otaku_tti.text_encoder import CLIPTextEncoder
from otaku_tti.vae import PretrainedVAE
from otaku_tti.diffusion import build_otaku_diffusion_unet
from otaku_tti.utils import save_image_grid

def parse_args():
    parser = argparse.ArgumentParser(description='Generate anime art with Otaku TTI Latent Diffusion Model')
    parser.add_argument('--prompt', '-p', type=str, default='1girl, solo, hatsune miku, twin braids, blue hair, teal eyes, anime screencap, masterpiece', help='Text prompt (Danbooru tags and/or natural language).')
    parser.add_argument('--negative-prompt', '-np', type=str, default='low quality, blurry, bad anatomy, deformed, monochrome, signature, watermark', help='Negative prompt for Classifier-Free Guidance.')
    parser.add_argument('--checkpoint', '-c', type=str, default='latest', help="Path to .pt checkpoint or 'latest'.")
    parser.add_argument('--output', '-o', type=str, default='./outputs/otaku_tti_diffusion/generated/output.png', help='Path to save the output image or grid.')
    parser.add_argument('--num-images', '-n', type=int, default=4, help='Number of images to generate.')
    parser.add_argument('--steps', type=int, default=25, help='Number of DPM-Solver++ denoising steps (default: 25).')
    parser.add_argument('--cfg', type=float, default=7.5, help='Classifier-Free Guidance scale (default: 7.5).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible generation.')
    parser.add_argument('--save-individual', action='store_true', help='Also save each generated image individually.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Compute device ('cuda' or 'cpu').")
    return parser.parse_args()

def resolve_checkpoint(ckpt_arg: str) -> Path:
    p = Path(ckpt_arg)
    if p.exists():
        return p
    for cand in [Path('./weights/otaku_tti_1.0_fp16.pt'), Path('./weights/otaku_tti_1.0.pt')]:
        if cand.exists():
            return cand
    ckpt_dir = Path('./outputs/otaku_tti_diffusion/checkpoints')
    if ckpt_arg.isdigit():
        cand = ckpt_dir / f'otaku_tti_diffusion_step_{int(ckpt_arg):07d}.pt'
        if cand.exists():
            return cand
    cand = ckpt_dir / 'otaku_tti_diffusion_latest.pt'
    if cand.exists():
        return cand
    return Path('./weights/otaku_tti_1.0.pt')

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print('=' * 64)
    print('  Otaku TTI - Anime Latent Diffusion Generator (1.0)')
    print('=' * 64)
    ckpt_path = resolve_checkpoint(args.checkpoint)
    config = {'clip_model': 'openai/clip-vit-base-patch32', 'vae_model': 'stabilityai/sd-vae-ft-mse', 'image_size': 256, 'latent_size': 32, 'cross_attention_dim': 512}
    unet = build_otaku_diffusion_unet(sample_size=config['latent_size'], cross_attention_dim=config['cross_attention_dim']).to(device=device, dtype=dtype)
    if ckpt_path.exists():
        print(f'[Otaku TTI] Loading checkpoint from {ckpt_path}...')
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if 'config' in ckpt:
            config.update(ckpt['config'])
        if 'ema' in ckpt:
            unet.load_state_dict(ckpt['ema'])
            print(f'[Otaku TTI] Loaded EMA weights (step {ckpt.get('step', 'unknown')}).')
        else:
            unet.load_state_dict(ckpt['unet'])
            print(f'[Otaku TTI] Loaded UNet weights (step {ckpt.get('step', 'unknown')}).')
    else:
        print(f'[Otaku TTI] Warning: Checkpoint not found at {ckpt_path}. Running with fresh weights.')
    unet.eval()
    print(f"[Otaku TTI] Loading CLIP text encoder '{config['clip_model']}'...")
    text_encoder = CLIPTextEncoder(model_name=config['clip_model'], device=device)
    print(f"[Otaku TTI] Loading VAE '{config['vae_model']}'...")
    vae = PretrainedVAE(model_name=config['vae_model'], device=device, dtype=dtype)
    scheduler = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', prediction_type='epsilon', algorithm_type='dpmsolver++', use_karras_sigmas=True)
    scheduler.set_timesteps(args.steps, device=device)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    prompts = [args.prompt] * args.num_images
    neg_prompts = [args.negative_prompt] * args.num_images
    _, cond_tokens, _ = text_encoder.encode(prompts)
    _, uncond_tokens, _ = text_encoder.encode(neg_prompts)
    cond_tokens = cond_tokens.to(device=device, dtype=dtype)
    uncond_tokens = uncond_tokens.to(device=device, dtype=dtype)
    text_embeddings = torch.cat([uncond_tokens, cond_tokens], dim=0)
    latent_size = config['latent_size']
    shape = (args.num_images, 4, latent_size, latent_size)
    latents = torch.randn(shape, generator=generator, device=device)
    latents = latents * scheduler.init_noise_sigma
    print(f'[Otaku TTI] Generating {args.num_images} image(s) with {args.steps} steps (CFG={args.cfg})...')
    t0 = time.time()
    use_amp = device.type == 'cuda'
    for step_idx, t in enumerate(scheduler.timesteps):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        with torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    t_unet = time.time() - t0
    images = vae.decode(latents)
    total_time = time.time() - t0
    print(f'[Otaku TTI] Denoising: {t_unet:.2f}s | Total: {total_time:.2f}s ({total_time / args.num_images:.2f}s/image)')
    save_image_grid(images, output_path, nrow=int(args.num_images ** 0.5) if args.num_images > 1 else 1)
    print(f'[Otaku TTI] Saved image -> {output_path}')
    if args.save_individual and args.num_images > 1:
        stem = output_path.stem
        ext = output_path.suffix
        for i in range(args.num_images):
            indiv_path = output_path.parent / f'{stem}_{i + 1:03d}{ext}'
            save_image_grid(images[i:i + 1], indiv_path, nrow=1)
        print(f'[Otaku TTI] Saved {args.num_images} individual images in {output_path.parent}')
if __name__ == '__main__':
    main()