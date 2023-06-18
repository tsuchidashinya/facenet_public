from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
import os
import time
import numpy as np
import pandas as pd


def cos_similarity(p1, p2): 
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))

    
log_file = open("log.txt", "w")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0,  min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,  device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
image_paths = ["tsuchida_3.JPG", "tsuchida_2.JPG", "mybrother_1.JPG", "mybrother_2.JPG", "angelina_1.jpeg", "angelina_2.jpeg", "will_smith_2.jpeg", "will_smith_1.jpeg"]
names = []
embeddings = []
for i, image_path in enumerate(image_paths):
    names.append(os.path.splitext(image_path)[0])
    image_path = os.path.join("images", image_path)
    img = Image.open(image_path)
    start = time.time()
    img_cropped = mtcnn(img)
    mtcnn_time = time.time()
    print(i, "processing_time(mtcnn - start): ", mtcnn_time - start, file=log_file)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    img_probs = img_probs.squeeze().to('cpu').detach().numpy().copy()
    end = time.time()
    print(i, "processing_time(resnet - mtcnn): ", end - mtcnn_time, file=log_file)
    print(file=log_file)
    embeddings.append(img_probs)
dists= [[cos_similarity(e1, e2) for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names), file=log_file)

