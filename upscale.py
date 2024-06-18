# %%
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image
import os
import threading
import shutil
import pandas as pd
import argparse


def process_dir(in_dir, out_dir):
    df = pd.read_csv(os.path.join(in_dir, "metadata.csv"))

    def get_text(file_name):
        row = df[df["file_name"] == file_name]
        return row["text"].values[0]

    src_dir = in_dir
    dst_dir = out_dir
    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
    ).to("cuda")
    os.makedirs(dst_dir, exist_ok=True)
    img_names = [
        img
        for img in sorted(os.listdir(src_dir))
        if img.endswith(".png") or img.endswith(".jpg")
    ]
    # Process images in batches
    batch_sz = 1
    for i in range(0, len(img_names), batch_sz):
        batch = img_names[i : i + batch_sz]
        prompts = [get_text(name) for name in batch]
        images = [Image.open(os.path.join(src_dir, name)) for name in batch]
        # Apply the pipeline to the batch
        batch_output = pipeline(
            prompt=prompts, image=images, noise_level=320, noise_level_sub=320
        )
        # Resize and save each image in the batch
        for idx, output in enumerate(batch_output.images):
            resized_image = output
            # resized_image = output.resize((512, 512))
            dst_img_path = os.path.join(dst_dir, batch[idx])
            resized_image.save(dst_img_path)
            print(f"Processed and saved: {batch[idx]}")


parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str)
parser.add_argument("--out_dir", type=str)

args = parser.parse_args()

process_dir(args.in_dir, args.out_dir)

import shutil

try:
    shutil.copyfile(
        os.path.join(args.in_dir, "metadata.csv"),
        os.path.join(args.out_dir, "metadata.csv"),
    )
except shutil.SameFileError:
    pass
# %%
