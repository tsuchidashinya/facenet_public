from facenet_pytorch import InceptionResnetV1
from common import device

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
resnet.classify = True



