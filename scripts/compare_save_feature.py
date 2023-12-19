import numpy as np
import os
import sys
sys.path.append('./')
sys.path.append('../')
from util.util import get_image_path_no_ext, cos_similarity
from network.facenet import Facenet
from options.option import args
import time

if not os.path.exists(args.out_dir):
   os.mkdir(args.out_dir)
image_paths = [os.path.join(args.input_dir, file_path) for file_path in os.listdir(args.input_dir)]
embeddings_dict = {}
facenet = Facenet()
for image_path in image_paths:
  try:
    img_feat = facenet.calc_face_feature(image_path)
    img_name = get_image_path_no_ext(image_path)
    np.save(os.path.join(args.out_dir, img_name + '.npy'), img_feat)
    embeddings_dict[str(img_name)] = img_feat
  except AttributeError:
    continue
  
np.save(os.path.join(args.out_dir, 'all.npy'), embeddings_dict)
start_bulk = time.time()
feature = np.load(os.path.join(args.out_dir, 'all.npy'), allow_pickle=True)
feature = feature.item()

for key in list(feature.keys()):
  cos_similarity(feature[key], img_feat)
end_bulk = time.time()
print(end_bulk - start_bulk)
for image_path in image_paths:
  try:
    img_name = get_image_path_no_ext(image_path)
    feature1 = np.load(os.path.join(args.out_dir, img_name + '.npy'), allow_pickle=True)
    cos_similarity(feature1, img_feat)
  except:
    continue
end_single = time.time()
print(end_single - end_bulk)



