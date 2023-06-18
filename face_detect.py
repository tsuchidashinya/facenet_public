from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os
import time
import numpy as np
import pandas as pd


def cos_similarity(p1, p2): 
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0,  min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,  device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
image_paths = ["naoki_1.JPG", "naoki_2.JPG", "shinya_1.JPG", "shinya_2.JPG", "tailor.jpeg", "tailor_1.jpeg"]
embeddings = []
for image_path in image_paths:
    img = Image.open(image_path)
    # start = time.time()
    img_cropped = mtcnn(img)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    img_probs = img_probs.squeeze().to('cpu').detach().numpy().copy()
    # end = time.time()
    embeddings.append(img_probs)
dists= [[cos_similarity(e1, e2) for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=image_paths, index=image_paths))

