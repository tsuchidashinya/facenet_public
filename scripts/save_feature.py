import numpy as np
import os
import sys
sys.path.append('./')
sys.path.append('../')
from util.util import get_image_path_no_ext
from network.facenet import Facenet
from options.option import args

image_paths = [os.path.join(args.input_dir, file_path) for file_path in os.listdir(args.input_dir)]
embeddings_dict = {}
facenet = Facenet()
for image_path in image_paths:
  try:
    print(image_path)
    img_feat = facenet.calc_face_feature(image_path)
    img_name = get_image_path_no_ext(image_path)
    embeddings_dict[str(img_name)] = img_feat
  except AttributeError:
    continue
  
np.save(args.out_file, embeddings_dict)



