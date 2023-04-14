import argparse
import shutil
import random
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Sample a subset of images from a given dataset')
parser.add_argument('--data-dir', type=str, help='path to the dataset directory')
parser.add_argument('--save-dir', type=str, help='path to the output directory')
parser.add_argument('--shots', type=int, help='number of images to sample per class')
args = parser.parse_args()

# Create the output directory if it doesn't exist
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# Get a list of all subdirectories in the dataset directory
subdirs = [d for d in Path(args.data_dir).iterdir() if d.is_dir()]

# Loop over each subdirectory and randomly select shots images to copy to the output directory
for subdir in tqdm(subdirs):
    class_name = subdir.name
    output_class_dir = Path(args.save_dir) / class_name
    if output_class_dir.exists():
        print(f'Warning: class {class_name} already exists in the output directory')
        raise ValueError
    output_class_dir.mkdir(parents=True, exist_ok=True)
    images = list(subdir.glob('*'))
    selected_images = random.sample(images, min(args.shots, len(images)))
    if len(selected_images) < args.shots:
        print(f'Warning: class {class_name} only has {len(selected_images)} images')
    for image in selected_images:
        dst_path = output_class_dir / image.name
        shutil.copy(image, dst_path)
