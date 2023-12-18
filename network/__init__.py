from PIL import Image
from mtcnn import mtcnn
from facenet import resnet


def calc_face_feature(img_path):
  img = Image.open(img_path)
  img_cropped = mtcnn(img)
  img_probs = resnet(img_cropped.unsqueeze(0))
  img_probs = img_probs.squeeze().to('cpu').detach().numpy().copy()