# Kepler ve Tess verilerini kullanarak model eğitmiş ve DR3 verilerini sınıflandırmıştık.
# Kepler-DR3 ve Tess-DR3 confusion matrixlerini çizer.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

labels = ['Detached', 'SemiDetached', 'OverContact', 'Ellipsoidal']

MergeKeplerTessGaiaUniquePredict = pd.read_csv('MergeKeplerTessGaiaUniquePredict.csv', index_col=False,
                                               dtype={"KeplerClass": "string", "KeplerMorph": "float", "TessClass": "string", "TessMorph": "float",
                                                      "DR3ClassPrediction": "string"})
# Kepler
MergeKeplerTessGaiaUniquePredictKeplerOnly = MergeKeplerTessGaiaUniquePredict[~np.isnan(MergeKeplerTessGaiaUniquePredict['KeplerMorph'])]

cm = confusion_matrix(MergeKeplerTessGaiaUniquePredictKeplerOnly['KeplerClass'],
                      MergeKeplerTessGaiaUniquePredictKeplerOnly['DR3ClassPrediction'], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('ConfusionMatrix_Kepler-DR3')
plt.ylabel('Kepler')
plt.xlabel('DR3')
plt.savefig('ConfusionMatrix_Kepler-DR3')
plt.close()

# Tess
MergeKeplerTessGaiaUniquePredictTessOnly = MergeKeplerTessGaiaUniquePredict[~np.isnan(MergeKeplerTessGaiaUniquePredict['TessMorph'])]

cm = confusion_matrix(MergeKeplerTessGaiaUniquePredictTessOnly['TessClass'],
                      MergeKeplerTessGaiaUniquePredictTessOnly['DR3ClassPrediction'], labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('ConfusionMatrix_Tess-DR3')
plt.ylabel('Tess')
plt.xlabel('DR3')
plt.savefig('ConfusionMatrix_Tess-DR3')
plt.close()
