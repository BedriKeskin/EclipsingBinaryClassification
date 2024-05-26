import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from tqdm import tqdm


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
    path_Algol = 'lcAlgol/Algol3070'
    Algol_files = [os.path.join(path_Algol, f) for f in os.listdir(path_Algol) if f.endswith('.txt')]

    path_Beta_Lyrae = 'lcBetaLyrae/BetaLyrae1070'
    Beta_Lyrae_files = [os.path.join(path_Beta_Lyrae, f) for f in os.listdir(path_Beta_Lyrae) if f.endswith('.txt')]

    path_W_UMa = 'lcWUMa/WUMa0470'
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


def load_lightCurvesVillanova():
    path_Detached = 'Detached'
    Detached_files = [os.path.join(path_Detached, f) for f in os.listdir(path_Detached) if f.endswith('.png')]

    path_Ellipsoidal = 'Ellipsoidal'
    Ellipsoidal_files = [os.path.join(path_Ellipsoidal, f) for f in os.listdir(path_Ellipsoidal) if f.endswith('.png')]

    path_OverCantact = 'OverContact'
    OverCantact_files = [os.path.join(path_OverCantact, f) for f in os.listdir(path_OverCantact) if f.endswith('.png')]

    path_SemiDetached = 'SemiDetached'
    SemiDetached_files = [os.path.join(path_SemiDetached, f) for f in os.listdir(path_SemiDetached) if
                          f.endswith('.png')]

    X = []
    label = []

    for file in Detached_files:
        label.append("Detached")
        X.append(file)

    for file in SemiDetached_files:
        label.append("SemiDetached")
        X.append(file)

    for file in OverCantact_files:
        label.append("OverContact")
        X.append(file)

    for file in Ellipsoidal_files:
        label.append("Ellipsoidal")
        X.append(file)

    return X, label


def load_lightCurvesVillanova_prediction():
    path_Detached = 'Detached_prediction'
    Detached_files = [os.path.join(path_Detached, f) for f in os.listdir(path_Detached) if f.endswith('.png')]

    path_Ellipsoidal = 'Ellipsoidal_prediction'
    Ellipsoidal_files = [os.path.join(path_Ellipsoidal, f) for f in os.listdir(path_Ellipsoidal) if f.endswith('.png')]

    path_OverCantact = 'OverContact_prediction'
    OverCantact_files = [os.path.join(path_OverCantact, f) for f in os.listdir(path_OverCantact) if f.endswith('.png')]

    path_SemiDetached = 'SemiDetached_prediction'
    SemiDetached_files = [os.path.join(path_SemiDetached, f) for f in os.listdir(path_SemiDetached) if
                          f.endswith('.png')]

    X = []
    label = []

    for file in Detached_files:
        label.append("Detached")
        X.append(file)

    for file in SemiDetached_files:
        label.append("SemiDetached")
        X.append(file)

    for file in OverCantact_files:
        label.append("OverContact")
        X.append(file)

    for file in Ellipsoidal_files:
        label.append("Ellipsoidal")
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


def image_feature(directory):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = []
    img_name = []
    for idx, i in enumerate(directory):
        print(idx, " image file: ", i)
        img = image.load_img(i)  # , target_size=(257, 194))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat = model.predict(x)
        feat = feat.flatten()
        features.append(feat)
        img_name.append(i)
    return features, img_name
