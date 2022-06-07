import cv2
import numpy as np
import dlib
import glob
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

data = []
labels = []

for i, imagePath in enumerate(glob.glob("C:\\Users\\pouya\\Desktop\\emotion detection\\train\\*\\*")):

  img = cv2.imread(imagePath)

  try:

    faces = detector(img, 1)

    for item in faces:

      (x1, y1) = item.left(), item.top()
      (x2, y2) = item.right(), item.bottom()

      roi = img[y1:y2, x1:x2]
      r_roi = cv2.resize(roi, (32, 32))

      data.append(r_roi)

      label = imagePath.split("\\")[-2]
      labels.append(label)

    if i % 1000 == 0:
      print(f"[INFO] {i}/29000 processed")  
  except:
    pass


le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data)/255

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

net = models.Sequential([
                         layers.Conv2D(64, (3, 3), activation='relu', input_shape = (32, 32, 3)),
                         layers.BatchNormalization(),
                         layers.Conv2D(64, (3, 3), activation='relu'),
                         layers.BatchNormalization(),

                         layers.MaxPool2D(),

                         layers.Conv2D(128, (3, 3), activation='relu', input_shape = (32, 32, 3)),
                         layers.BatchNormalization(),
                         layers.Conv2D(64, (3, 3), activation='relu'),
                         layers.BatchNormalization(),

                         layers.MaxPool2D(),

                         layers.Flatten(),
                         layers.Dense(32, activation='relu'),
                         layers.BatchNormalization(),
                         layers.Dense(7, activation='softmax')
                        ])

net.compile(loss = 'categorical_crossentropy',
            optimizer = 'sgd',
            metrics = ['accuracy'])

H = net.fit(X_train, y_train, batch_size=16, epochs=30, validation_data = (X_test, y_test))


net.save("emotion_model.h5")

plt.style.use('ggplot')
plt.plot(H.history["accuracy"], label = 'train accuracy')
plt.plot(H.history["val_accuracy"], label = 'test accuracy')
plt.plot(H.history["loss"], label = 'train loss')
plt.plot(H.history["val_loss"], label = 'test loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Accuracy/Loss')
plt.show()