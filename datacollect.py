import dlib
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

dataset_dir = 'datasetfaces'
nameID = str(input("Diga seu nome: ")).upper()
path = os.path.join(dataset_dir, nameID)

exists = os.path.exists(path)

if exists:
    print("Nome ja cadastrado!")
    nameID = str(input("Diga seu nome novamente: ")).upper()
else:
    os.makedirs(path)

# Carregar o detector de faces do dlib
detector = dlib.get_frontal_face_detector()

# Carregar o modelo de reconhecimento facial do dlib pré-treinado
model_file = 'dlib_face_recognition_resnet_model_v1.dat'
face_recognizer = dlib.face_recognition_model_v1(model_file)

# Carregar o modelo de predição do dlib
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Função para extrair os descritores faciais de uma imagem
def extract_face_descriptors(image):
    dets = detector(image, 1)
    face_descriptors = []
    for det in dets:
        shape = predictor(image, det)
        face_descriptor = face_recognizer.compute_face_descriptor(image, shape)
        face_descriptors.append(face_descriptor)
    return face_descriptors

# Carregar o arquivo de descritores faciais existentes (se houver)
descriptors_file = os.path.join(path, 'descriptors.npy')
if os.path.isfile(descriptors_file):
    existing_descriptors = np.load(descriptors_file)
else:
    existing_descriptors = []

# Captura de vídeo
video = cv2.VideoCapture(0)

count = len(existing_descriptors)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecção de faces
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Extração dos descritores faciais
        face_descriptors = extract_face_descriptors(frame)

        # Adicionar os novos descritores faciais aos descritores existentes
        existing_descriptors.extend(face_descriptors)

        # Salvar os descritores faciais em um arquivo
        np.save(descriptors_file, existing_descriptors)

        # Desenhar retângulo ao redor da face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Coletando...", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


        count += 1

    cv2.imshow("Coleta de dados", frame)
    cv2.waitKey(1)

    if count >= 100:
        break

video.release()
cv2.destroyAllWindows()

print("Treinando o modelo com a face de {} .......".format(nameID))
descriptors_file = os.path.join(path, 'descriptors.npy')
descriptors = np.load(descriptors_file)

# Gerar os rótulos correspondentes aos descritores faciais
labels = np.full((descriptors.shape[0],), nameID)
labels_file = os.path.join(path, 'labels.npy')
np.save(labels_file, labels)

# Carregar os descritores e rótulos de outros indivíduos (opcional)
outros_descritores = []
outros_rotulos = []
for other_nameID in os.listdir(dataset_dir):
    if other_nameID != nameID and os.path.isdir(os.path.join(dataset_dir, other_nameID)):
        other_path = os.path.join(dataset_dir, other_nameID)
        other_descriptors_file = os.path.join(other_path, 'descriptors.npy')
        other_labels_file = os.path.join(other_path, 'labels.npy')
        if os.path.exists(other_descriptors_file) and os.path.exists(other_labels_file):
            other_descriptors = np.load(other_descriptors_file)
            other_labels = np.load(other_labels_file)
            outros_descritores.append(other_descriptors)
            outros_rotulos.append(other_labels)

# Combinar os descritores e rótulos de todos os indivíduos, se existirem
if outros_descritores and outros_rotulos:
    outros_descritores = np.concatenate(outros_descritores, axis=0)
    outros_rotulos = np.concatenate(outros_rotulos, axis=0)
    descritores = np.concatenate((descriptors, outros_descritores), axis=0)
    rotulos = np.concatenate((labels, outros_rotulos), axis=0)
else:
    descritores = descriptors
    rotulos = labels

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(descritores, rotulos, test_size=0.2, random_state=42)

# Inicializar o modelo de classificação (neste exemplo, utilizamos SVM)
model = SVC(C=0.1, kernel= 'poly', gamma='scale')


curr_dir = os.getcwd()


try:
    # Treinar o modelo
    model.fit(X_train, y_train)
    # Realizar previsões
    y_pred = model.predict(X_test)
    # Salvar em .pkl
    model_file = os.path.join(curr_dir, 'modelo_treinado.pkl')
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    # Calcular a precisão do modelo
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Precisao do modelo:", accuracy)
except:
    print("modelo nao treinado devido a quantia de classses ser inferior a 2")





