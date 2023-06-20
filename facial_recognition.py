import dlib
import cv2
import numpy as np
import pickle

# Carregar o detector de faces do dlib
detector = dlib.get_frontal_face_detector()

# Carregar o modelo de reconhecimento facial do dlib pré-treinado
model_file = 'dlib_face_recognition_resnet_model_v1.dat'
face_recognizer = dlib.face_recognition_model_v1(model_file)

# Carregar o modelo de predição do dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Carregar o modelo treinado
modelo_treinado_file = 'modelo_treinado.pkl'
with open(modelo_treinado_file, 'rb') as f:
    modelo_treinado = pickle.load(f)

# Função para extrair os descritores faciais de uma imagem
def extract_face_descriptors(image):
    dets = detector(image, 1)
    face_descriptors = []
    for det in dets:
        shape = predictor(image, det)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_descriptors.append(face_descriptor)
    return face_descriptors

# Captura de vídeo
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção de faces
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extração dos descritores faciais
        face_descriptors = extract_face_descriptors(frame)

        # Realizar previsões
        if len(face_descriptors) > 0:
            descriptors = np.array(face_descriptors)
            predictions = modelo_treinado.predict(descriptors)
            cv2.putText(frame, predictions[0], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "Nao reconhecido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Desenhar retângulo ao redor da face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


    cv2.imshow("Reconhecimento Facial", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
