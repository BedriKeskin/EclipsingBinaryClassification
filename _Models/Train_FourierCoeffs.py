import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import models

Kepler = pd.read_csv("/Users/tiga/Documents/EclipsingBinaryClassification/Kepler/FourierCoeffs_Kepler_3xP.csv",
                     delim_whitespace=False, index_col=False)
Kepler2 = Kepler.loc[:, 'a0':'label']

Tess = pd.read_csv("/Users/tiga/Documents/EclipsingBinaryClassification/Tess/FourierCoeffs_TESS_3xP.csv",
                   delim_whitespace=False, index_col=False)
Tess2 = Tess.loc[:, 'a0':'label']

Kepler2Tess2 = pd.concat([Kepler2, Tess2], ignore_index=True)

X = Kepler2Tess2.drop('label', axis=1).values
y = Kepler2Tess2['label']

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

# prediction
file = '/Users/tiga/Documents/EclipsingBinaryClassification/Gaia/FourierCoeffs_Gaia_3xP.csv'
Gaia = pd.read_csv(file)
Gaia2 = Gaia.loc[:, 'a0':'w']
array = Gaia2.to_numpy()
array_reshaped = array.reshape(-1, array.shape[1], 1)
predictions = model.predict(array_reshaped)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_classes)
Gaia.loc[:, 'prediction'] = predicted_labels

Gaia.to_csv(file, index=False)

# Plot Confusion Matrix
Gaia = Gaia[Gaia['Villanova Class'].notna()]  # Sınıfı belli olmayanları gözardı et

labels = ['Detached', 'SemiDetached', 'OverContact', 'Ellipsoidal']

cm = confusion_matrix(Gaia['Villanova Class'], Gaia['prediction'], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

accuracy = accuracy_score(Gaia['Villanova Class'], Gaia['prediction'])
precision = precision_score(Gaia['Villanova Class'], Gaia['prediction'], pos_label='positive', average='micro')
recall = recall_score(Gaia['Villanova Class'], Gaia['prediction'], pos_label='positive', average='micro')
f1 = f1_score(Gaia['Villanova Class'], Gaia['prediction'], pos_label='positive', average='micro')

print(f'accuracy: {accuracy}')
print(f'precision: {precision}')
print(f'recall: {recall}')
print(f'F1 score: {f1}')

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('ConfusionMatrix_FourierCoeffs_3xP')
plt.ylabel('Kepler&TESS')
plt.xlabel('DR3')
plt.savefig('ConfusionMatrix_FourierCoeffs_3xP')
plt.close()
