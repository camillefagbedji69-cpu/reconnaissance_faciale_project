from mtcnn import MTCNN
from keras_facenet import FaceNet

def get_embedding(face):
  embedder = FaceNet()
  face = face.astype('float32')
  face = np.expand_dims(face, axis = 0)
  embeding = embedder.embeddings(face)

  return embeding[0]


detector = MTCNN()

def extract_face(filename, size = (160,160)) :
  img = cv2.imread(filename)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  results = detector.detect_faces(img)

  if len (results) == 0 :
    return None
  x1, y1, w, h = results[0]['box']
  x1, y1 = abs(x1), abs(y1)
  face = img[y1:y1+h, x1:x1+w]
  face = cv2.resize(face, size)
  return face