from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import os
from tqdm import tqdm
import shutil
import csv
import argparse

import torch

device = torch.device("cuda:0")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map={"": 0})

def get_blip_caption(filename):
    image = Image.open(filename)
    text = "An illustration of"

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)

    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("filename", filename)
    print("generated_text", generated_text)
    return generated_text

parser = argparse.ArgumentParser(description='')
parser.add_argument('--in_dir', type=str)
parser.add_argument('--out_dir', type=str)
args = parser.parse_args()
processed_dir = args.in_dir
captioned_dir = args.out_dir
# shutil.copytree(processed_dir, captioned_dir, dirs_exist_ok=True)

# Define the directory to start the search
for src_dir in os.listdir(captioned_dir):
    print(f"processing {src_dir}...")
    metadata_file = os.path.join(captioned_dir, "metadata.csv")
    with open(metadata_file, 'w') as file:
        writer = csv.writer(file)
        # Write the header of the CSV file
        writer.writerow(['file_name', 'text'])
        # Loop through root, dirs, and files in the directory
        for filename in tqdm(sorted(os.listdir(os.path.join(captioned_dir)))):
            # Check if the file is a png file
            if filename.endswith(".png"):
                png_file_path = os.path.join(captioned_dir, filename)
                # Call get_blip_caption function with the png file path
                caption = get_blip_caption(png_file_path)
                writer.writerow([filename, caption])


print("Processing Complete")
