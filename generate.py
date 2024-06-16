import torch
import os
import argparse

from diffusers import StableDiffusionKDiffusionPipeline

def process_dir_standard(in_dir, out_dir, seed=123456):
    os.makedirs(out_dir, exist_ok=True)
    pipe = StableDiffusionKDiffusionPipeline.from_pretrained(in_dir).to('cuda')
    pipe.set_scheduler('sample_dpmpp_2m')
    for idx, prompt in enumerate(COMMON_PROMPTS):
        img_name = str(idx).zfill(4) + ".png"
        out_path = os.path.join(out_dir, img_name)
        image = pipe(prompt=prompt,
            num_inference_steps=50, generator=torch.Generator().manual_seed(seed), 
            width=768, height=768, 
            use_karras_sigmas=True).images[0]
        image.save(out_path)

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--prompts', type=str)
args = parser.parse_args()

with open(args.prompts, 'r') as file:
    COMMON_PROMPTS = file.readlines()
COMMON_PROMPTS = [prompt.strip() for prompt in COMMON_PROMPTS]

process_dir_standard(args.in_dir, args.out_dir)