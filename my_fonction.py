from mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
import numpy as np
def get_embedding(face):
  embedder = FaceNet()
  face = face.astype('float32')
  face = np.expand_dims(face, axis = 0)
  embeding = embedder.embeddings(face)

  return embeding[0]

detector = MTCNN()

def extract_face(file):
    img = Image.open(file).convert("RGB")
    pixels = np.asarray(img)

    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None

    x1, y1, w, h = results[0]["box"]
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + w, y1 + h

    face = pixels[y1:y2, x1:x2]
    face = Image.fromarray(face).resize((160, 160))
    return np.asarray(face)




