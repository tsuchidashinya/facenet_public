from facenet_pytorch import InceptionResnetV1
from facenet_pytorch import MTCNN
import torch
from PIL import Image


class Facenet:
  def __init__(self):
    device = torch.device('cpu')
    self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, thresholds=[
              0.6, 0.6, 0.7], factor=0.709, post_process=True, device=device)
    self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    self.resnet.classify = True

  def calc_face_feature(self, img_path):
    img = Image.open(img_path)
    img_cropped = self.mtcnn(img)
    img_probs = self.resnet(img_cropped.unsqueeze(0))
    img_probs = img_probs.squeeze().to('cpu').detach().numpy().copy()
    return img_probs



