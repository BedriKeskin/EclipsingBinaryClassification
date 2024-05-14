# -*- coding: utf-8 -*-

import itertools
import os
import sys
import datetime

import keras
# import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import Helpers

sys.stdout = open("NN" + ".out", 'w')
now1 = datetime.datetime.now()
print(now1)

X, label = Helpers.load_lightCurvesTxt()

# Etiketleri kategorik hale getir
encoder = LabelEncoder()
label = encoder.fit_transform(label)
# numpy arraye çevir
X = np.array(X)
label = np.array(label)

# Özellik ölçeklendirme
X_ss = StandardScaler().fit_transform(X)
# X_ss = StandardScaler().transform(X)

# run an LDA and use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit(X_ss, label).transform(X_ss)
print('Original number of features:', X_ss.shape[1])
print('Reduced number of features:', X_lda.shape[1])

# X_lda = X  # Feature extraction iptali için bunu uncomment et
# Verileri eğitim ve test kümelerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X_lda, label, test_size=0.8, random_state=42)
X_test1, X_test2, y_test1, y_test2 = train_test_split(X_test, y_test, test_size=0.2, random_state=42)


# # NN
from keras.layers import Dense
from keras.models import Sequential

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))  # txt
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(3, activation='softmax'))

loss = 'sparse_categorical_crossentropy'
optimizer = 'adam'
# %100
# NN sonu

# # NN2
# from keras.layers import Dense, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam
#
# model = Sequential()
# model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
# model.add(Dropout(0.2))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
#
# loss = 'binary_crossentropy'
# optimizer = Adam(learning_rate=0.001)
# # 65.96%  # Not: Dense(3, activation='softmax') ve loss = 'sparse_categorical_crossentropy' yapınca bu da 100% veriyor
# # NN2 sonu


model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test1, y_test1))
loss, accuracy = model.evaluate(X_test1, y_test1, verbose=0)
print("Model loss: {:.2f}%".format(loss * 100))
model.save('NN_model.h5')

# # RF
# from sklearn.ensemble import RandomForestClassifier
# model = RandomForestClassifier()

# # SVM
# from sklearn.svm import SVC
# model = SVC(kernel='rbf', gamma='scale', C=1.0)
#
# history = model.fit(X_train, y_train)
# accuracy = model.score(X_test1, y_test1)

print("Model accuracy: {:.2f}%".format(accuracy * 100))

# Training ve validation grafiği
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.plot(range(10), history.history["accuracy"], label='Training Accuracy')
plt.plot(range(10), history.history["val_accuracy"], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel("Epoch number")
plt.subplot(1, 2, 2)
plt.plot(range(10), history.history["loss"], label='Training Loss')
plt.plot(range(10), history.history["val_loss"], label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel("Epoch number")
plt.savefig('NN_TraininAndValidation')
plt.close()

predictions = model.predict(X_test2)

file2wr = open("NN" + "_predictions" + ".txt", "w")
file2wr.write("True\tPredicted\n")
for idx, x in enumerate(y_test2):
    str2wr = "{:}\t{:}\n".format(x, predictions[idx])
    file2wr.write(str2wr)
file2wr.close()

confusion_matrix = confusion_matrix(y_test2, pd.DataFrame(predictions).idxmax(axis=1))  # NN için
# confusion_matrix = confusion_matrix(y_test2, predictions)  # SVM, RF için
class_names = ['Algol', 'B Lyr', 'W UMa']
# Confusion matrix grafiği
plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.get_cmap('Blues'))
plt.title("NN" + " Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], ".2f"), horizontalalignment="center",
             color="white" if confusion_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.savefig("NN" + "_ConfusionMatrix")
plt.close()

now2 = datetime.datetime.now()
print(now2)
print(now2 - now1)
