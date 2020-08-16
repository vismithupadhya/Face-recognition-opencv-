import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle   
import os
import imutils
import time
import datetime

curr_path = os.getcwd()   #returns the current working directory

print("Loading face detection model")
model = r'E:\face rec\model'
proto_path = os.path.join(curr_path, model, 'deploy.prototxt')
model_path = os.path.join(curr_path, model, 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, model, 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)
database = r'E:\face rec\database'

data_base_path = os.path.join(curr_path, database)

filenames = []
for path, subdirs, files in os.walk(data_base_path):
    for name in files:
        filenames.append(os.path.join(path, name))

face_embeddings = []
face_names = []

for (i, filename) in enumerate(filenames):
    print("Processing image {}".format(filename))

    image = cv2.imread(filename)
    image = imutils.resize(image, width=600)

    (h, w) = image.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    i = np.argmax(face_detections[0, 0, :, 2])
    confidence = face_detections[0, 0, i, 2]

    if confidence >= 0.5:

        box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        face = image[startY:endY, startX:endX]

        face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0), True, False)

        face_recognizer.setInput(face_blob)
        face_recognitions = face_recognizer.forward()

        name = filename.split(os.path.sep)[-2]

        face_embeddings.append(face_recognitions.flatten())
        face_names.append(name)

data = {"embeddings": face_embeddings, "names": face_names}

le = LabelEncoder()
labels = le.fit_transform((data["names"]))

recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open('recognizer.pickle', "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open("le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
print("Training complete")


curr_path = os.getcwd()

print("Loading face detection model")
model = r'E:\face rec\model'
proto_path = os.path.join(curr_path, model, 'deploy.prototxt')
model_path = os.path.join(curr_path, model, 'res10_300x300_ssd_iter_140000.caffemodel')
face_detector = cv2.dnn.readNetFromCaffe(prototxt=proto_path, caffeModel=model_path)

print("Loading face recognition model")
recognition_model = os.path.join(curr_path, model, 'openface_nn4.small2.v1.t7')
face_recognizer = cv2.dnn.readNetFromTorch(model=recognition_model)

recognizer = pickle.loads(open('recognizer.pickle', "rb").read())
le = pickle.loads(open('le.pickle', "rb").read())

print("Starting test video file")
vs = cv2.VideoCapture(0)
time.sleep(1)

while True:

    ret, frame = vs.read()
    frame = imutils.resize(frame, width=600)

    (h, w) = frame.shape[:2]

    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)

    face_detector.setInput(image_blob)
    face_detections = face_detector.forward()

    for i in range(0, face_detections.shape[2]):
        confidence = face_detections[0, 0, i, 2]

        if confidence >= 0.5:
            box = face_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            face_blob = cv2.dnn.blobFromImage(face, 1.0/255, (96, 96), (0, 0, 0), True, False)

            face_recognizer.setInput(face_blob)
            vec = face_recognizer.forward()

            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            text = "{}: {:.2f}".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()
