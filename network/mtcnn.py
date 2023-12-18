from facenet_pytorch import MTCNN
from common import device

mtcnn = MTCNN(image_size=140, margin=0, min_face_size=20, thresholds=[
              0.6, 0.6, 0.7], factor=0.709, post_process=True, device=device)