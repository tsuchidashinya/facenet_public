import pandas as pd
import yaml
from facenet.util import cos_similarity, get_image_path_no_ext
from facenet.facenet import calc_face_feature
from facenet.args import args


image_paths = []
with open(args.yaml_file, 'r') as yaml_file:
    image_paths = yaml.safe_load(yaml_file)

names = []
embeddings = []
for i, image_path in enumerate(image_paths):
    image_name = get_image_path_no_ext(image_path)
    names.append(image_name)
    img_probs = calc_face_feature(image_path)
    embeddings.append(img_probs)

dists = [[cos_similarity(e1, e2) for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names), file=log_file)
print(pd.DataFrame(dists, columns=names, index=names))
