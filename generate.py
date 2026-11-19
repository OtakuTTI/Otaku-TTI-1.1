import argparse
import json
import os
from pathlib import Path
import time
from typing import List, Optional
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from diffusers import DPMSolverMultistepScheduler
from otaku_tti.text_encoder import CLIPTextEncoder
from otaku_tti.vae import PretrainedVAE
from otaku_tti.diffusion import build_otaku_diffusion_unet
from otaku_tti.utils import save_image_grid

def load_tag_map(map_path: Optional[Path]=None) -> dict:
    if map_path is None:
        map_path = Path(__file__).resolve().parent / 'tag_map_ja.json'
    if map_path.exists():
        try:
            with open(map_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {'初音ミク': 'hatsune miku', 'ミク': 'hatsune miku', 'アスカ': 'asuka langley', '霊夢': 'reimu hakurei', 'レム': 'rem (re:zero)', 'セーラームーン': 'sailor moon', '悟空': 'goku', 'ナルト': 'naruto uzumaki', 'ルフィ': 'monkey d. luffy', 'ツインテール': 'twintails', '笑顔': 'smiling', '青髪': 'blue hair', '赤髪': 'red hair', '金髪': 'blonde hair', '黒髪': 'black hair', '制服': 'school uniform', '猫耳': 'cat ears', '最高品質': 'masterpiece, best quality', '少女': '1girl', '少年': '1boy', '風景': 'scenery, landscape', '人物なし': 'no humans'}

def translate_prompt(prompt: str, tag_map: dict) -> str:
    translated = prompt
    for ja, en in tag_map.items():
        if ja in translated:
            translated = translated.replace(ja, en)
    return translated

def parse_args():
    parser = argparse.ArgumentParser(description='Generate 512x512 anime art with Otaku TTI 1.1')
    parser.add_argument('--prompt', '-p', type=str, default='1girl, solo, hatsune miku, teal twintails, smiling, anime screencap, masterpiece, 512x512', help='Prompt (Danbooru tags, English, or Japanese).')
    parser.add_argument('--negative-prompt', '-np', type=str, default='low quality, blurry, bad anatomy, deformed, monochrome, text, watermark', help='Negative prompt for CFG.')
    parser.add_argument('--checkpoint', '-c', type=str, default=None, help='Path to .pt weights file.')
    parser.add_argument('--output', '-o', type=str, default='output.png', help='Output image path.')
    parser.add_argument('--num-images', '-n', type=int, default=1, help='Number of images to generate.')
    parser.add_argument('--steps', '-s', type=int, default=25, help='Denoising steps (default: 25).')
    parser.add_argument('--cfg', type=float, default=7.5, help='Classifier-Free Guidance scale (default: 7.5).')
    parser.add_argument('--seed', type=int, default=None, help='Random seed.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda or cpu).')
    return parser.parse_args()

def resolve_weights(ckpt_arg: Optional[str]) -> Path:
    if ckpt_arg:
        p = Path(ckpt_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f'Checkpoint not found at: {ckpt_arg}')
    candidates = [Path(__file__).resolve().parent / 'weights' / 'otaku_tti_1.1_fp16.pt', Path(__file__).resolve().parent / 'weights' / 'otaku_tti_1.1.pt', Path('weights/otaku_tti_1.1_fp16.pt'), Path('weights/otaku_tti_1.1.pt')]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError('No Otaku TTI 1.1 weights found in weights/ directory.')

@torch.inference_mode()
def main():
    args = parse_args()
    tag_map = load_tag_map()
    prompt = translate_prompt(args.prompt, tag_map)
    if prompt != args.prompt:
        print(f'[Otaku TTI 1.1] Translated Japanese prompt: {args.prompt} -> {prompt}')
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print('=' * 64)
    print('  Otaku TTI 1.1 - 512x512 Anime Latent Diffusion Generator')
    print(f'  Device: {device} | Precision: {dtype}')
    print('=' * 64)
    ckpt_path = resolve_weights(args.checkpoint)
    print(f'[Otaku TTI 1.1] Loading weights from: {ckpt_path.name}...')
    config = {'clip_model': 'openai/clip-vit-base-patch32', 'vae_model': 'stabilityai/sd-vae-ft-mse', 'latent_size': 64, 'cross_attention_dim': 512}
    unet = build_otaku_diffusion_unet(sample_size=config['latent_size'], cross_attention_dim=config['cross_attention_dim'])
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict) and 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif isinstance(ckpt, dict) and 'unet' in ckpt:
        state_dict = ckpt['unet']
    else:
        state_dict = ckpt
    unet.load_state_dict(state_dict)
    del ckpt, state_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    unet = unet.to(device=device, dtype=dtype)
    unet.eval()
    print('[Otaku TTI 1.1] Loading CLIP & VAE...')
    text_encoder = CLIPTextEncoder(model_name=config['clip_model'], device=device)
    vae = PretrainedVAE(model_name=config['vae_model'], device=device)
    scheduler = DPMSolverMultistepScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', algorithm_type='dpmsolver++', solver_order=2)
    scheduler.set_timesteps(args.steps, device=device)
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        print(f'[Otaku TTI 1.1] Seed: {args.seed}')
    prompts = [prompt] * args.num_images
    neg_prompts = [args.negative_prompt] * args.num_images
    _, cond_tokens, _ = text_encoder.encode(prompts)
    _, uncond_tokens, _ = text_encoder.encode(neg_prompts)
    cond_tokens = cond_tokens.to(device=device, dtype=dtype)
    uncond_tokens = uncond_tokens.to(device=device, dtype=dtype)
    text_embeddings = torch.cat([uncond_tokens, cond_tokens], dim=0)
    latent_size = config['latent_size']
    shape = (args.num_images, 4, latent_size, latent_size)
    latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma
    print(f'[Otaku TTI 1.1] Generating {args.num_images} 512x512 image(s) with {args.steps} steps (CFG={args.cfg})...')
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
    images = vae.decode(latents.float())
    total_time = time.time() - t0
    print(f'[Otaku TTI 1.1] Denoising: {t_unet:.2f}s | Total: {total_time:.2f}s ({total_time / args.num_images:.2f}s/image)')
    save_image_grid(images, output_path, nrow=int(args.num_images ** 0.5) if args.num_images > 1 else 1)
    print(f'[Otaku TTI 1.1] Saved 512x512 output -> {output_path}')
if __name__ == '__main__':
    main()
