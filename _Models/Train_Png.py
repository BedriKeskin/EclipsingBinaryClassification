import datetime

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split

import data
import models

now1 = datetime.datetime.now()
print("Time start: ", now1)

Images, Labels = data.ImagesLabelsInFolder("../Kepler/PNG-")
Images1, Labels1 = data.ImagesLabelsInFolder("../Tess/PNG-")

Images += Images1
Labels += Labels1

images = np.array(Images)
labels = np.array(Labels)
labels = to_categorical(labels, 4)

print(images.shape)
print(labels.shape)

# Split Data (Train, Test, Validation)
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.33, random_state=77)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.75, random_state=77)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)  # 'val_accuracy'

checkpoint = ModelCheckpoint(
    filepath='Vgg19_Berk_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'.keras',
    monitor='val_accuracy',
    verbose=1,
    save_best_only=True
)

model = models.vgg19_Berk()
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, verbose=1,
                    callbacks=[lr_scheduler, checkpoint, earlystop])  # , batch_size=256 ,

test = model.evaluate(x_test, y_test)
print("test loss, test acc:", test)

"""
# Prediction
ImagesAndLabelsPrediction = []
ImagesAndLabelsPrediction += data.ImagesAndLabelsPredictInFolder("Keplerx")
ImagesAndLabelsPrediction += data.ImagesAndLabelsPredictInFolder("Tessx")

imagesPrediction = np.array(ImagesAndLabelsPrediction[0])
labelsPrediction = np.array(ImagesAndLabelsPrediction[1])

prediction = mymodel.evaluate(imagesPrediction, labelsPrediction)
print("prediction loss, prediction acc:", prediction)
"""
now2 = datetime.datetime.now()
print("Time end: ", now2)
print("Duration: ", now2 - now1)

# modeli yeniden eğitince eski hali ile karşılaştırma
# np.testing.assert_allclose(
#    model.predict(test_input), reconstructed_model.predict(test_input)
# )
