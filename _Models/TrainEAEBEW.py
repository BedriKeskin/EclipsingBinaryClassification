import datetime

import keras
from keras.api.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

import globals
import models

now1 = datetime.datetime.now()
print("Time start: ", now1)

X, y = globals.PrepareXy("../Kepler/Alkim-/")

X_train, X_validationtest, y_train, y_validationtest = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
X_validation, X_test, y_validation, y_test = train_test_split(X_validationtest, y_validationtest, test_size=0.4, random_state=43)
print(X_validation.shape)
print(X_test.shape)

modelName = 'EAEBEW_vgg19_2_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.keras'

checkpoint = ModelCheckpoint(
    filepath=modelName,
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True
)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
model = models.vgg19_2(3)
history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), epochs=10, verbose=1, callbacks=[checkpoint, earlystop])  # , batch_size=256 ,

results = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(results)

model = keras.saving.load_model(modelName)
results = model.evaluate(X_test, y_test)
print(model.metrics_names)
print(results)

now2 = datetime.datetime.now()
print("Time end: ", now2)
print("Duration: ", now2 - now1)
