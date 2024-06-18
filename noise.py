from skimage import img_as_ubyte
import os
from tqdm import tqdm
import os
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage import img_as_ubyte, img_as_float32
import argparse


def gaussian_noise(img, stddev):
    var = stddev * stddev
    noisy_img = random_noise(img, mode="gaussian", var=var, clip=True)
    return noisy_img


def process_image(img_path, img_out_path, method, strength):
    # Load the image
    img = img_as_float32(imread(img_path))
    img_processed = method(img, strength)
    img_processed = img_as_ubyte(img_processed)
    imsave(img_out_path, img_processed)


def process_images(in_dir, out_dir, method, strength):
    os.makedirs(out_dir, exist_ok=True)
    for img_name in tqdm(os.listdir(in_dir)):
        if not img_name.endswith(".png"):
            continue
        img_path = os.path.join(in_dir, img_name)
        img_out_path = os.path.join(out_dir, img_name)
        process_image(img_path, img_out_path, method, strength)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--in_dir",
    type=str,
)
parser.add_argument(
    "--out_dir",
    type=str,
)
parser.add_argument(
    "--gaussian_noise",
    type=float,
)

args = parser.parse_args()

process_images(args.in_dir, args.out_dir, gaussian_noise, args.gaussian_noise)

import shutil

shutil.copyfile(
    os.path.join(args.in_dir, "metadata.csv"),
    os.path.join(args.out_dir, "metadata.csv"),
)
