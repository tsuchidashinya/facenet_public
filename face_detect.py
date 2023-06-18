from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os
import time
import numpy as np
import pandas as pd
import yaml
import argparse


pd.set_option('display.max_rows', 150)
pd.set_option('display.max_columns', 20)
parser = argparse.ArgumentParser()
parser.add_argument('-y', '--yaml_file', type=str, default='load_image_list.yaml')
args = parser.parse_args()
image_paths = []
with open(args.yaml_file, 'r') as yaml_file:
    image_paths = yaml.safe_load(yaml_file)



def cos_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


log_file = open("log.txt", "w")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0,  min_face_size=20, thresholds=[
              0.6, 0.7, 0.7], factor=0.709, post_process=True,  device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

names = []
embeddings = []
for i, image_path in enumerate(image_paths):
    image_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(image_name)[0]
    names.append(image_name_no_ext)
    img = Image.open(image_path)
    start = time.time()
    img_cropped = mtcnn(img)
    mtcnn_time = time.time()
    print(i, "processing_time(mtcnn - start): ",
          mtcnn_time - start, file=log_file)
    print(i, "processing_time(mtcnn - start): ", mtcnn_time - start)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    img_probs = img_probs.squeeze().to('cpu').detach().numpy().copy()
    end = time.time()
    print(i, "processing_time(resnet - mtcnn): ",
          end - mtcnn_time, file=log_file)
    print(i, "processing_time(resnet - mtcnn): ", end - mtcnn_time)
    print(file=log_file)
    embeddings.append(img_probs)
dists = [[cos_similarity(e1, e2) for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names), file=log_file)
print(pd.DataFrame(dists, columns=names, index=names))
