import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--yaml_file', type=str, default='yaml/sample_images.yaml')
parser.add_argument('--input_dir', type=str, default='sample/image')
parser.add_argument('--feature_file', type=str, default='sapmle_data.npy')
parser.add_argument('--image_file', type=str, default='sample/test/test.jpg')
parser.add_argument('--out_dir', type=str)
parser.add_argument('--out_file', type=str)
args = parser.parse_args()