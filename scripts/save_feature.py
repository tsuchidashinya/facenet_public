import numpy as np
import os
from util import get_image_path_no_ext
from network import calc_face_feature
from option import args

image_paths = [os.path.join(args.input_dir, file_path) for file_path in os.listdir(args.input_dir)]
embeddings_dict = {}
for image_path in image_paths:
  try:
    img_feat = calc_face_feature(image_path)
    img_name = get_image_path_no_ext(image_path)
    embeddings_dict[str(img_name)] = img_feat
  except AttributeError:
    continue
  
np.save('sample_data.npy', embeddings_dict)



