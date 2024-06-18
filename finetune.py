import subprocess
import threading
import argparse
import shutil
import os
import tempfile


def create_tmp_train_dir(in_dir):
    # Create a temporary directory
    tmp_dir = tempfile.mkdtemp()

    # Create subdirectories 'train' and 'val' inside the temporary directory
    train_dir = os.path.join(tmp_dir, "train")
    val_dir = os.path.join(tmp_dir, "val")
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    # Copy all files from in_dir to the 'train' subdirectory
    for filename in os.listdir(in_dir):
        src_file = os.path.join(in_dir, filename)
        dst_file = os.path.join(train_dir, filename)
        if os.path.isfile(src_file):
            shutil.copy(src_file, dst_file)

    return tmp_dir


def run_command_sd(in_dir, out_dir, orig_in_dir, num_processes=1):
    resolution = 512
    command = f"""accelerate launch --num_processes={num_processes} diffusers_repo/examples/text_to_image/train_text_to_image.py \
--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
--dataset_name="{in_dir}" \
--use_ema \
--resolution={resolution} \
--center_crop \
--random_flip \
--train_batch_size=4 \
--max_train_steps=2000 \
--gradient_accumulation_steps=1  \
--gradient_checkpointing \
--learning_rate=5e-06 \
--max_grad_norm=1 \
--lr_scheduler="constant" \
--lr_warmup_steps=0 \
--output_dir="{out_dir}" \
--report_to="wandb" \
--validation_prompts "a astronaut riding a horse by nulevoy" "a candle next to a mirror by nulevoy" "a car by nulevoy" "a cheese phone by nulevoy" "a tree with a light shining on it by nulevoy" "a mountain landscape by nulevoy" "a village in a thunderstorm by nulevoy" \
--validation_epochs=50 \
--seed=123456 \
--token="nulevoy" \
--tracker_project_name="{orig_in_dir.replace("/", "_")}"
"""
    subprocess.run(command, shell=True)


def process_directories(root, directories, method):
    while directories:
        # Start two threads if there are at least two directories left
        threads = []
        for i in args.gpu_ids:
            if not directories:
                break
            dir = directories.pop()
            print(f"Processing {dir}", flush=True)
            cuda_device = i
            thread = threading.Thread(target=method, args=(cuda_device, dir))
            thread.start()
            threads.append(thread)

        # Wait for both threads to complete
        for thread in threads:
            thread.join()


parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str)
parser.add_argument("--out_dir", type=str)
parser.add_argument("--num_processes", type=int, default=1)
args = parser.parse_args()

tmp_train_dir = create_tmp_train_dir(args.in_dir)
run_command_sd(tmp_train_dir, args.out_dir, args.in_dir, args.num_processes)
shutil.rmtree(tmp_train_dir)
