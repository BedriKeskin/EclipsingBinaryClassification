import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import os


def plot_confusion_matrix(cnn, input_data, input_labels):
    # Compute flare predictions for the test dataset
    predictions = cnn.predict(input_data)

    # Convert to binary classification
    predictions = (predictions > 0.5).astype('int32')

    # Compute the confusion matrix by comparing the test labels (ds.test_labels) with the test predictions
    cm = metrics.confusion_matrix(input_labels, predictions, labels=[0, 1])
    cm = cm.astype('float')

    # Normalize the confusion matrix results.
    cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.matshow(cm_norm, cmap='binary_r')

    plt.title('Confusion matrix', y=1.08)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Flare', 'No Flare'])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Flare', 'No Flare'])

    plt.xlabel('Predicted')
    plt.ylabel('True')

    fmt = '.2f'
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, format(cm_norm[i, j], fmt),
                    ha="center", va="center", color="white" if cm_norm[i, j] < thresh else "black")
    plt.show()


def load_lightCurvesTxt():
    path_Algol = '/Users/tiga/My Drive/Doktora/Truba/lcAlgol/Algol3070'
    Algol_files = [os.path.join(path_Algol, f) for f in os.listdir(path_Algol) if f.endswith('.txt')]

    path_Beta_Lyrae = '/Users/tiga/My Drive/Doktora/Truba/lcBetaLyrae/BetaLyrae1070'
    Beta_Lyrae_files = [os.path.join(path_Beta_Lyrae, f) for f in os.listdir(path_Beta_Lyrae) if f.endswith('.txt')]

    path_W_UMa = '/Users/tiga/My Drive/Doktora/Truba/lcWUMa/WUMa0470'
    W_UMa_files = [os.path.join(path_W_UMa, f) for f in os.listdir(path_W_UMa) if f.endswith('.txt')]

    X = []
    label = []

    for file in Algol_files:
        label.append("Algol")
        fileContent = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x, y = line.strip().split('\t')
                # fileContent.append([float(x), float(y)])
                fileContent.append(float(y))
        X.append(fileContent)

    for file in Beta_Lyrae_files:
        label.append("B Lyr")
        fileContent = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x, y = line.strip().split('\t')
                fileContent.append(float(y))
        X.append(fileContent)

    for file in W_UMa_files:
        label.append("W UMa")
        fileContent = []
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                x, y = line.strip().split('\t')
                fileContent.append(float(y))
        X.append(fileContent)

    return X, label


def load_lightCurvesPng():
    path_Algol = 'lcAlgol/Algol3070'
    Algol_files = [os.path.join(path_Algol, f) for f in os.listdir(path_Algol) if f.endswith('.png')]

    path_Beta_Lyrae = 'lcBetaLyrae/BetaLyrae1070'
    Beta_Lyrae_files = [os.path.join(path_Beta_Lyrae, f) for f in os.listdir(path_Beta_Lyrae) if f.endswith('.png')]

    path_W_UMa = 'lcWUMa/WUMa0470'
    W_UMa_files = [os.path.join(path_W_UMa, f) for f in os.listdir(path_W_UMa) if f.endswith('.png')]

    X = []
    label = []

    for file in Algol_files:
        label.append("Algol")
        X.append(file)

    for file in Beta_Lyrae_files:
        label.append("B Lyr")
        X.append(file)

    for file in W_UMa_files:
        label.append("W UMa")
        X.append(file)

    return X, label


def load_predictPng():
    path_Algol = 'lcAlgol/Algol_testLC'
    Algol_files = [os.path.join(path_Algol, f) for f in os.listdir(path_Algol) if f.endswith('.png')]

    path_Beta_Lyrae = 'lcBetaLyrae/BetaLyrae_testLC'
    Beta_Lyrae_files = [os.path.join(path_Beta_Lyrae, f) for f in os.listdir(path_Beta_Lyrae) if f.endswith('.png')]

    path_W_UMa = 'lcWUMa/WUMa_testLC'
    W_UMa_files = [os.path.join(path_W_UMa, f) for f in os.listdir(path_W_UMa) if f.endswith('.png')]

    X = []
    label = []

    for file in Algol_files:
        label.append("Algol")
        X.append(file)

    for file in Beta_Lyrae_files:
        label.append("B Lyr")
        X.append(file)

    for file in W_UMa_files:
        label.append("W UMa")
        X.append(file)

    return X, label


def plotlc(lcname, outdir, t, ph, mag, P):
    lcbasename = lcname.split('/')[-1]
    plt.figure(1)
    plt.subplot(211)
    plt.gca().invert_yaxis()
    tcorr = t - t[0]
    plt.plot(tcorr, mag, 'bo', markersize=0.5)
    plt.ylabel('magnitude')
    plt.title(lcbasename + ' P=' + str(P))
    plt.xlabel('time - ' + str(t[0]))
    plt.subplot(212)
    plt.gca().invert_yaxis()
    plt.plot(ph, mag, 'bo', markersize=0.5)
    plt.ylabel('magnitude')
    plt.xlabel('phase')
    plt.savefig(outdir + '/' + lcbasename + '.png', format="png")
    plt.close()
