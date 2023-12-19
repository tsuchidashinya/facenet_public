import os
import shutil
import sys
sys.path.append('./')
sys.path.append('../')
from options.option import args
from util import extract_file
import random


if not os.path.exists(args.out_dir):
   os.mkdir(args.out_dir)
file_list, _ = extract_file(args.input_dir, 0, 10000)
random.shuffle(file_list)
count = 0
for i in range(500):
  count += 1
  out_path = os.path.join(args.out_dir, 'img_No' + str(count) + '.jpg')
  if os.path.exists(out_path):
     continue
  shutil.copy(file_list[i], out_path)




      