# https://medium.com/mlearning-ai/image-classification-for-beginner-a6de7a69bc78
import datetime

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import regularizers
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, \
    recall_score
from sklearn.model_selection import train_test_split

import Helpers

now1 = datetime.datetime.now()
print("Başlangıç zamanı: ", now1)

X, label = Helpers.load_lightCurvesVillanova()
df_all = pd.DataFrame({'img': X, 'label': label})
df = df_all.sample(n=100)
df = df_all  # sample'ı iptal etmek için bu satırı aç
print("İmaj sayısı: ", len(df.index))

df_labels = {
    'Detached': 0,
    'SemiDetached': 1,
    'OverContact': 2,
    'Ellipsoidal': 3
}
# Encode
df['encode_label'] = df['label'].map(df_labels)

X = []
for img in df['img']:
    img = cv2.imread(str(img))
    # img = augment_function(img)
    # img = cv2.resize(img, (96, 96))
    # img = img / 255
    X.append(img)

y = df['encode_label']

# input_shape = plt.imread(df['img'][0]).shape

X_train, X_test_val, y_train, y_test_val = train_test_split(X, y)
print("Train imaj sayısı: ", len(X_train))
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val)
print("Validation imaj sayısı: ", len(X_val))
print("Test imaj sayısı: ", len(X_test))

# # model 1
# base_model = VGG16(include_top=False, weights='imagenet', input_shape=(194, 257, 3))
#
# for layer in base_model.layers:
#     layer.trainable = False
# base_model.layers[-2].trainable = True
# base_model.layers[-3].trainable = True
# base_model.layers[-4].trainable = True
# base_model.summary()

# model = Sequential()
# model.add(InputLayer(input_shape=(194, 257, 3)))
# model.add(base_model)
# model.add(Flatten())
# model.add(Dropout(0.2))
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.2))
# # model.add(Dense(len(df_labels), activation='softmax'))  # 4
# model.add(Dense(4, activation='softmax'))  # 4
# # model.summary()
#
# model.compile(
#     optimizer="adam",
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy'])
# #

# model https://github.com/burakulas/ebclass Burak Ulaş https://ui.adsabs.harvard.edu/abs/2023arXiv230602686U/abstract
l1 = 32
l2 = 32
l3 = 64
l4 = 128
l5 = 256

lrate = 1e-4

model = tf.keras.models.Sequential([
    # First convolution
    tf.keras.layers.Conv2D(l1, (3, 3), activation='relu',
                           kernel_regularizer=regularizers.l2(0.001),
                           input_shape=(194, 257, 3), padding='same'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Second convolution
    tf.keras.layers.Conv2D(l2, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Third convolution
    tf.keras.layers.Conv2D(l3, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Fourth convolution
    tf.keras.layers.Conv2D(l4, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Fifth convolution
    tf.keras.layers.Conv2D(l5, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=regularizers.l2(0.001)
                           ),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Flatten and Dropout
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    # Hidden layer
    tf.keras.layers.Dense(l5, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
opt = keras.optimizers.Adam(learning_rate=lrate)
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#

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
model.save('ml_Villanova_TESS.h5')

# Training ve validation grafiği
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
print(history.history)
plt.plot(range(epochs), history.history["accuracy"], label='Training Accuracy')
plt.plot(range(epochs), history.history["val_accuracy"], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Villanova_TESS TESS Training and Validation Accuracy')
plt.xlabel("Epoch number")
plt.subplot(1, 2, 2)
plt.plot(range(epochs), history.history["loss"], label='Training Loss')
plt.plot(range(epochs), history.history["val_loss"], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Villanova_TESS TESS Training and Validation Loss')
plt.xlabel("Epoch number")
plt.savefig('ml_Villanova_TESS_TrainingAndValidation')
plt.close()

X_predict, label_predict = Helpers.load_lightCurvesVillanova_prediction()
df_predict_all = pd.DataFrame({'img': X_predict, 'label': label_predict})
df_predict = df_predict_all.sample(n=200)
df_predict = df_predict_all  # sample'ı iptal etmek için bu satırı aç
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

X_predict2 = tf.stack(X_predict2)

label_predict = tf.stack(df_predict['encode_label'])

loss, accuracy = model.evaluate(X_predict2, label_predict)
print("Prediction loss: {:.2f}%".format(loss * 100))
print("Prediction accuracy: {:.2f}%".format(accuracy * 100))

predictions = model.predict(X_predict2)
df_predict['prediction_Detached'] = predictions[:, 0]
df_predict['prediction_SemiDetached'] = predictions[:, 1]
df_predict['prediction_OverContact'] = predictions[:, 2]
df_predict['prediction_Ellipsoidal'] = predictions[:, 3]
df_predict.to_csv(r'ml_Villanova_TESS_prediction.txt', header=True, index=True, sep='\t', mode='w')

prediction = np.argmax(predictions, axis=-1)
cm = confusion_matrix(df_predict['encode_label'], prediction)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df_labels)
disp.plot()
plt.title('ml_Villanova_TESS_ConfusionMatrix')
plt.savefig('ml_Villanova_TESS_ConfusionMatrix')
plt.close()

print("f1_score ", f1_score(df_predict['encode_label'], prediction, average='micro'))
print("accuracy_score ", accuracy_score(df_predict['encode_label'], prediction))
print("precision_score ", precision_score(df_predict['encode_label'], prediction, average='micro'))
print("recall_score ", recall_score(df_predict['encode_label'], prediction, average='micro'))

now2 = datetime.datetime.now()
print("Bitiş zamanı: ", now2)
print("Geçen süre: ", now2 - now1)
