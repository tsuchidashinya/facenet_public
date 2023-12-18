import os
import shutil
from option import args
from util import extract_file


if not os.path.exists(args.out_dir):
   os.mkdir(args.out_dir)
file_list, _ = extract_file(args.input_dir, 0, 500)
print(file_list)
print(len(file_list))
count = 0
for file_path in file_list:
  count += 1
  out_path = os.path.join(args.out_dir, 'img_No' + str(count) + '.jpg')
  if os.path.exists(out_path):
     continue
  shutil.copy(file_path, out_path)




      