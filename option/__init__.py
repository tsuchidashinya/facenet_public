import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--yaml_file', type=str, default='yaml/sample_images.yaml')
parser.add_argument('-i', '--input_dir', type=str, default='sample/image')
parser.add_argument('-f', '--file', type=str, default='sapmle_data.npy')

parser.add_argument('-o', '--out_dir', type=str)
args = parser.parse_args()