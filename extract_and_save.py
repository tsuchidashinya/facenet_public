import os
import argparse
import shutil


def extract_file(dir_path, count, LIMIT):
  local_count = count
  file_list = []
  for f in os.listdir(dir_path):
    if local_count > LIMIT:
        break
    file_path = os.path.join(dir_path, f)
    if os.path.isfile(file_path):
      local_count += 1
      file_list.append(file_path)
    else:
      append_file, new_count = extract_file(file_path, local_count, LIMIT)
      local_count = new_count
      file_list.extend(append_file)
  return file_list, local_count

def get_image_path(image_path):
    image_name = os.path.basename(image_path)
    return image_name


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir', type=str)
parser.add_argument('-o', '--out_dir', type=str)
args = parser.parse_args()

if not os.path.exists(args.out_dir):
   os.mkdir(args.out_dir)
file_list, _ = extract_file(args.dir, 0, 500)
print(file_list)
print(len(file_list))
count = 0
for file_path in file_list:
  count += 1
  out_path = os.path.join(args.out_dir, 'img_No' + str(count) + '.jpg')
  if os.path.exists(out_path):
     continue
  shutil.copy(file_path, out_path)




      