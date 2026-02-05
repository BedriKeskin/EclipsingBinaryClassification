import os
import pathlib
import shutil
import warnings
import joblib
import random
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    concatenate, Dropout, BatchNormalization,
    RandomFlip, RandomTranslation, RandomZoom
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import regularizers


# --- 1. SEED SABİTLEME ---
def reset_random_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)


reset_random_seeds()

# Ayarlar
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')


# --- Callback: Makro F1 Skoru ---
class MacroF1Callback(Callback):
    def __init__(self, x_val, y_val, batch_size=1024):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_probs = self.model.predict(self.x_val, batch_size=self.batch_size, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        y_true = np.argmax(self.y_val, axis=1)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        logs['val_f1_macro'] = f1
        print(f" - val_f1_macro: {f1:.4f}")


# --- 2. VERİ YÜKLEME (TEK KLASÖR + DEĞİŞKENLİ) ---
print("\nAdım 1: Veri Yükleme ve Ön İşleme...")

# --- AYARLAR VE SABİTLER ---
CSV_PATH = '../MowlaviModelParams.csv'
DATA_DIRS = ['TrainValidation']
TEST_DIR = 'Test'
IMG_HEIGHT, IMG_WIDTH = 128, 96
CLASSES = ['EA', 'EB', 'EW']
TARGET_MODEL_TYPE = 'TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1'

TABULAR_COLUMNS = ['frequency', 'geom_model_reference_level', 'geom_model_gaussian1_phase',
                   'geom_model_gaussian1_sigma', 'geom_model_gaussian1_depth',
                   'geom_model_gaussian2_phase', 'geom_model_gaussian2_sigma',
                   'geom_model_gaussian2_depth', 'geom_model_cosine_half_period_amplitude',
                   'geom_model_cosine_half_period_phase']

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV dosyası bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)
df.set_index('source_id', inplace=True)

# Hızlı Erişim için Sözlük (Dictionary)
print(f"Tablo verisi hafızaya alınıyor (Hedef Model: {TARGET_MODEL_TYPE})...")
cols_to_fetch = TABULAR_COLUMNS + ['model_type']
params_dict = df[cols_to_fetch].to_dict('index')

image_data, tabular_data, labels = [], [], []

for data_dir in DATA_DIRS:
    if not os.path.exists(data_dir):
        print(f"[UYARI] Klasör bulunamadı: {data_dir}")
        continue

    print(f"\nKlasör taranıyor: {data_dir}")
    for category in CLASSES:
        path = os.path.join(data_dir, category)
        if not os.path.exists(path): continue

        image_files = [f for f in os.listdir(path) if f.endswith('.png')]
        print(f"  -> '{category}' sınıfından {len(image_files)} resim yükleniyor...")

        for img_name in image_files:
            try:
                source_id = int(os.path.splitext(img_name)[0])
                row_data = params_dict.get(source_id)

                if row_data is None: continue

                # DEĞİŞKEN KULLANIMI BURADA YAPILDI
                if row_data['model_type'] != TARGET_MODEL_TYPE: continue

                star_params = [row_data[col] for col in TABULAR_COLUMNS]
                img_path = os.path.join(path, img_name)

                # Resim yükleme
                img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
                img_array = img_to_array(img) / 255.0

                image_data.append(img_array)
                tabular_data.append(star_params)
                labels.append(category)
            except Exception:
                continue

image_data = np.array(image_data)
tabular_data = np.array(tabular_data)
print(f"\nToplam Yüklenen Veri Sayısı: {len(image_data)}")

# Etiketleme
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(labels)
y = to_categorical(integer_labels)

# --- 3. BÖLME VE SCALING ---
print("\nAdım 2: Veri Bölme ve Ölçekleme...")
(X_img_train, X_img_val, X_tabular_train, X_tabular_val, y_train, y_val) = train_test_split(
    image_data, tabular_data, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_tabular_train = scaler.fit_transform(X_tabular_train)
X_tabular_val = scaler.transform(X_tabular_val)
joblib.dump(scaler, "tabular_scaler.joblib")

# --- 4. MODEL MİMARİSİ (WITH AUGMENTATION) ---
print("\nAdım 3: Model Mimarisi (Data Augmentation Aktif)...")

# --- Görüntü Kolu (CNN) ---
image_input = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3), name='image_input')

# Augmentation Katmanları
x = RandomFlip("horizontal")(image_input)
x = RandomTranslation(height_factor=0.05, width_factor=0.05)(x)
x = RandomZoom(height_factor=0.05)(x)

# 1. Konvolüsyon Bloğu
cnn = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D((2, 2))(cnn)
cnn = Dropout(0.25)(cnn)

# 2. Konvolüsyon Bloğu
cnn = Conv2D(64, (3, 3), activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D((2, 2))(cnn)
cnn = Dropout(0.3)(cnn)

# 3. Konvolüsyon Bloğu
cnn = Conv2D(128, (3, 3), activation='relu', padding='same')(cnn)
cnn = BatchNormalization()(cnn)
cnn = MaxPooling2D((2, 2))(cnn)
cnn = Dropout(0.4)(cnn)

cnn = Flatten()(cnn)
cnn_output = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(cnn)

# --- Tablo Kolu (MLP) ---
tabular_input = Input(shape=(len(TABULAR_COLUMNS),), name='tabular_input')

mlp = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(tabular_input)
mlp = BatchNormalization()(mlp)
mlp = Dropout(0.4)(mlp)

mlp_output = Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.005))(mlp)

# --- Birleştirme ---
combined = concatenate([cnn_output, mlp_output])

x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.005))(combined)
x = Dropout(0.5)(x)

output = Dense(len(CLASSES), activation='softmax', name='output')(x)

model = Model(inputs=[image_input, tabular_input], outputs=output)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# --- 5. EĞİTİM ---
print("\nAdım 4: Model Eğitimi...")

macro_f1_cb = MacroF1Callback(
    x_val=[X_img_val, X_tabular_val],
    y_val=y_val,
    batch_size=1024
)

checkpoint_cb = ModelCheckpoint(
    filepath="LCandMP_best_f1.keras",
    monitor="val_f1_macro",
    save_best_only=True,
    mode="max",
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_f1_macro",
    patience=15,
    mode="max",
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    x=[X_img_train, X_tabular_train], y=y_train,
    validation_data=([X_img_val, X_tabular_val], y_val),
    epochs=18,
    batch_size=64,
    callbacks=[macro_f1_cb, checkpoint_cb, earlystop_cb, reduce_lr_cb],
    verbose=1
)

# --- 6. GRAFİK ---
fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(history.history['loss'], label='Train Loss', color='tab:red', linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', color='tab:orange', linewidth=2, linestyle='--')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='tab:red')
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(history.history['accuracy'], label='Train Acc', color='tab:blue', linewidth=2)
ax2.plot(history.history['val_accuracy'], label='Val Acc', color='tab:cyan', linewidth=2, linestyle='--')
if 'val_f1_macro' in history.history:
    ax2.plot(history.history['val_f1_macro'], label='Val F1', color='green', linestyle=':')

ax2.set_ylabel('Accuracy / F1', color='tab:blue')

plt.title(f'Learning Curve ({TARGET_MODEL_TYPE})')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
plt.savefig("LCandMP_learning_curve.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 7. TEST ---
print("\nAdım 5: Test Değerlendirmesi...")
X_img_test, X_tabular_test, labels_test = [], [], []
test_image_paths = []

for category in CLASSES:
    path = os.path.join(TEST_DIR, category)
    if not os.path.exists(path): continue

    image_files = [f for f in os.listdir(path) if f.endswith('.png')]
    print(f"Test - '{category}' sınıfı işleniyor...")

    for img_name in image_files:
        try:
            source_id = int(os.path.splitext(img_name)[0])
            row_data = params_dict.get(source_id)
            if row_data is None: continue

            # DEĞİŞKEN KULLANIMI TEST İÇİN DE EKLENDİ
            if row_data['model_type'] != TARGET_MODEL_TYPE: continue

            star_params = [row_data[col] for col in TABULAR_COLUMNS]
            img_path = os.path.join(path, img_name)
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
            img_array = img_to_array(img) / 255.0

            X_img_test.append(img_array)
            X_tabular_test.append(star_params)
            labels_test.append(category)
            test_image_paths.append(img_path)
        except:
            continue

X_img_test = np.array(X_img_test)
X_tabular_test = np.array(X_tabular_test)
X_tabular_test = scaler.transform(X_tabular_test)
integer_labels_test = label_encoder.transform(labels_test)
y_test = to_categorical(integer_labels_test)

y_pred_probs = model.predict([X_img_test, X_tabular_test], batch_size=512, verbose=0)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

acc = accuracy_score(y_true_classes, y_pred_classes)
class_report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_,
                                     zero_division=0)

print(f"\nTest Accuracy: {acc:.4f}")
print("Classification Report:\n", class_report)

with open("LCandMP_results.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n\n")
    f.write(class_report)

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("LCandMP_confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.show()

# --- 8. KLASÖRLEME ---
print("\nAdım 6: Hatalı/Doğru Klasörleme...")
output_cm_dir = pathlib.Path('TestCM_LCandMP')
class_names = label_encoder.classes_

if output_cm_dir.exists():
    shutil.rmtree(output_cm_dir)
output_cm_dir.mkdir(parents=True, exist_ok=True)

for i, src_path in enumerate(test_image_paths):
    true_cls = class_names[y_true_classes[i]]
    pred_cls = class_names[y_pred_classes[i]]
    target_folder = output_cm_dir / f"{true_cls}-{pred_cls}"
    target_folder.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(src_path, target_folder)
    except:
        pass

print("Tüm işlemler tamamlandı.")