import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import models

Kepler = pd.read_csv("/Users/tiga/Documents/EclipsingBinaryClassification/Kepler/FourierCoeffs_Kepler.csv",
                     delim_whitespace=False, index_col=False)
Kepler = Kepler.drop('ID', axis=1)
Kepler = Kepler.drop('morph', axis=1)
Kepler = Kepler.drop('T0', axis=1)
Kepler = Kepler.drop('P', axis=1)

Tess = pd.read_csv("/Users/tiga/Documents/EclipsingBinaryClassification/Tess/FourierCoeffs_TESS.csv",
                   delim_whitespace=False, index_col=False)
Tess = Tess.drop('ID', axis=1)
Tess = Tess.drop('morph', axis=1)
Tess = Tess.drop('T0', axis=1)
Tess = Tess.drop('P', axis=1)

df = pd.concat([Kepler, Tess], ignore_index=True)

X = df.drop('label', axis=1).values
y = df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

model = models.sequential2(X_train)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train.reshape(-1, X_train.shape[1], 1), y_train, epochs=20, batch_size=32,
          validation_data=(X_test.reshape(-1, X_test.shape[1], 1), y_test))

loss, accuracy = model.evaluate(X_test.reshape(-1, X_test.shape[1], 1), y_test)
print(f'loss: {loss}, accuracy: {accuracy}')
