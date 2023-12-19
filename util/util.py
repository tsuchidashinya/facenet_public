import numpy as np
import os

def cos_similarity(p1, p2):
    return np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))


def get_image_path_no_ext(image_path):
    image_name = os.path.basename(image_path)
    image_name_no_ext = os.path.splitext(image_name)[0]
    return image_name_no_ext

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