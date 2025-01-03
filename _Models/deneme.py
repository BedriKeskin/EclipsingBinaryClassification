import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import globals


KeplerTessGaia = pd.read_csv('./KeplerTessGaia_Predict_Sequential.csv', index_col=False,
                             dtype={"KIC": "string", "TIC": "string", "DR3": "string"})

true_labels = KeplerTessGaia['KeplerClass'].fillna('') + KeplerTessGaia['TessClass'].fillna('')
predicted_class_names = KeplerTessGaia['DR3ClassPrediction']

# accuracy = accuracy_score(true_labels, predicted_class_names)
# precision = precision_score(true_labels, predicted_class_names, pos_label='positive', average='micro')
# recall = recall_score(true_labels, predicted_class_names, pos_label='positive', average='micro')
# f1 = f1_score(true_labels, predicted_class_names, average='weighted')

accuracy = accuracy_score(true_labels, predicted_class_names)
precision = precision_score(true_labels, predicted_class_names, average='macro', zero_division=0)
recall = recall_score(true_labels, predicted_class_names, average='macro', zero_division=0)
f1 = f1_score(true_labels, predicted_class_names, average='macro', zero_division=0)

print('accuracy: ', accuracy)
print('precision: ', precision)
print('recall: ', recall)
print('F1 score: ', f1)

labels = globals.GetLabels(globals.Roche)

cm = confusion_matrix(true_labels, predicted_class_names, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix_-_Gaia')
plt.savefig('ConfusionMatrix_Berk_Gaia_deneme')