import numpy as np
from annoy import AnnoyIndex
import sys
sys.path.append('./')
sys.path.append('../')
from options.option import args
from network.facenet import Facenet
from util import cos_similarity
import cv2


feature = np.load(args.feature_file, allow_pickle=True)
feature = feature.item()
t = AnnoyIndex(list(feature.values())[0].shape[0])
key_array = []
for i, key in enumerate(list(feature.keys())):
  t.add_item(i, feature[key])
  key_array.append(key)
t.build(10)
facenet = Facenet()
test_feat = facenet.calc_face_feature(args.image_file)
r, d = t.get_nns_by_vector(test_feat, 5, include_distances=True)
for index, i in enumerate(r):
  img = cv2.imread('sample/image/' + str(key_array[i]) + '.jpg')
  result = cos_similarity(feature[key_array[i]], test_feat)
  # if index > 3:
  #   cv2.imshow(str(key_array[i]), img)
  print(str(key_array[i]), result)
  
test_img = cv2.imread(args.image_file)
# cv2.imshow('test', test_img)
# cv2.waitKey(10000)

