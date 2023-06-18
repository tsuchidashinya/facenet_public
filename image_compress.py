from PIL import Image
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--yaml_file', type=str, default='compress_image_list.yaml')
args = parser.parse_args()
image_paths = []
with open(args.yaml_file, 'r') as yaml_file:
    image_paths = yaml.safe_load(yaml_file)

for i, image_path in enumerate(image_paths):
    img = Image.open(image_path)
    width, height = img.size
    bairitu = 3
    width2 = width / bairitu
    height2 = height / bairitu
    img2 = img.resize((int(width2), int(height2)))
    img2.save(image_path)