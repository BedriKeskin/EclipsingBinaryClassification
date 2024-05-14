# https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78

import pandas as pd
import Helpers
import datetime

now1 = datetime.datetime.now()
print("Başlangıç zamanı: ", now1)

X, label = Helpers.load_lightCurvesPng()
df_all = pd.DataFrame({'img': X, 'label': label})
#df = df_all.sample(n=1000)
df = df_all
print("İmaj sayısı: ", len(df.index))

df_labels = {
    'Algol': 0,
    'B Lyr': 1,
    'W UMa': 2
}
# Encode
df['encode_label'] = df['label'].map(df_labels)

import cv2

X = []
for img in df['img']:
    img = cv2.imread(str(img))
    # img = augment_function(img)
    # img = cv2.resize(img, (96, 96))
    # img = img / 255
    X.append(img)

y = df['encode_label']

import matplotlib.pyplot as plt

# input_shape = plt.imread(df['img'][0]).shape

from sklearn.model_selection import train_test_split

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
print("Train imaj sayısı: ", len(X_train))
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)
print("Validation imaj sayısı: ", len(X_val))
print("Test imaj sayısı: ", len(X_test))

from keras.applications.vgg16 import VGG16

base_model = VGG16(include_top=False, weights='imagenet', input_shape=(194, 257, 3))

for layer in base_model.layers:
    layer.trainable = False
base_model.layers[-2].trainable = True
base_model.layers[-3].trainable = True
base_model.layers[-4].trainable = True
# base_model.summary()

from keras.models import *
from keras.layers import *

model = Sequential()
model.add(InputLayer(input_shape=(194, 257, 3)))
model.add(base_model)
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
# model.add(Dense(len(df_labels), activation='softmax'))  # 3
model.add(Dense(3, activation='softmax'))  # 3
# model.summary()

model.compile(
    optimizer="adam",
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

import tensorflow as tf

X_train = tf.stack(X_train)
y_train = tf.stack(y_train)

X_val = tf.stack(X_val)
y_val = tf.stack(y_val)

epochs = 5
history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val))

X_test = tf.stack(X_test)
y_test = tf.stack(y_test)

loss, accuracy = model.evaluate(X_test, y_test)
print("Model loss: {:.2f}%".format(loss * 100))
print("Model accuracy: {:.2f}%".format(accuracy * 100))
model.save('ml_png.h5')

# Training ve validation grafiği
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(range(epochs), history.history["acc"], label='Training Accuracy')
plt.plot(range(epochs), history.history["val_acc"], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epoch number")
plt.subplot(1, 2, 2)
plt.plot(range(epochs), history.history["loss"], label='Training Loss')
plt.plot(range(epochs), history.history["val_loss"], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("Epoch number")
plt.savefig('ml_png_TraininAndValidation')
plt.close()

X_predict, label_predict = Helpers.load_predictPng()
df_predict_all = pd.DataFrame({'img': X_predict, 'label': label_predict})
# df_predict = df_predict_all.sample(n=5000)
df_predict = df_predict_all
print("Prediction imaj sayısı: ", len(df_predict.index))


# Encode
df_predict['encode_label'] = df_predict['label'].map(df_labels)

X_predict2 = []
for img in df_predict['img']:
    img = cv2.imread(str(img))
    # img = augment_function(img)
    # img = cv2.resize(img, (96, 96))
    # img = img / 255
    X_predict2.append(img)

encode_label = df_predict['encode_label']

X_predict2 = tf.stack(X_predict2)

# label_predict = tf.stack(label_predict)

predictions = model.predict(X_predict2)
df_predict['prediction_Algol'] = predictions[:, 0]
df_predict['prediction_BLyr'] = predictions[:, 1]
df_predict['prediction_WUMa'] = predictions[:, 2]
df_predict.to_csv(r'ml_png_prediction.txt', header=True, index=True, sep='\t', mode='a')

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, \
    recall_score
import numpy as np

prediction = np.argmax(predictions, axis=-1)
cm = confusion_matrix(df_predict['encode_label'], prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df_labels)
disp.plot()
plt.title('ml_png_ConfusionMatrix')
plt.savefig('ml_png_ConfusionMatrix')
plt.close()

print("f1_score ", f1_score(df_predict['encode_label'], prediction, average='micro'))
print("accuracy_score ", accuracy_score(df_predict['encode_label'], prediction))
print("precision_score ", precision_score(df_predict['encode_label'], prediction, average='micro'))
print("recall_score ", recall_score(df_predict['encode_label'], prediction, average='micro'))

now2 = datetime.datetime.now()
print("Bitiş zamanı: ", now2)
print("Geçen süre: ",now2 - now1)

