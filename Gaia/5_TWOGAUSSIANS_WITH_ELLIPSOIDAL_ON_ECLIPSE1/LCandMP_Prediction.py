import glob
import os
import random
import shutil

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model


SRC_DIR = "PlotModel"
TARGET_DIR = "AutoClassified"
CSV_PATH = "../MowlaviModelParams.csv"
MODEL_PATH = "LCandMP_best_f1.keras"
SCALER_PATH = "tabular_scaler.joblib"

SAMPLE_COUNT = 100000
RANDOM_SEED = 42

CLASSES = ['EA', 'EB', 'EW']
IMG_HEIGHT, IMG_WIDTH = 128, 96  # eğitimde kullandığın boyut (height, width)
TABULAR_COLUMNS = ['frequency', 'geom_model_reference_level', 'geom_model_gaussian1_phase', 'geom_model_gaussian1_sigma', 'geom_model_gaussian1_depth', 'geom_model_gaussian2_phase', 'geom_model_gaussian2_sigma', 'geom_model_gaussian2_depth', 'geom_model_cosine_half_period_amplitude', 'geom_model_cosine_half_period_phase'] # all parameters

# ============================================

def base_no_ext(path):
    return os.path.splitext(os.path.basename(path))[0].strip()

def safe_move(src_path, dest_dir):
    """Hedefte aynı isim varsa _1, _2 ... ekleyerek taşır."""
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(src_path)
    stem, ext = os.path.splitext(name)
    dest_path = os.path.join(dest_dir, name)
    i = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(dest_dir, f"{stem}_{i}{ext}")
        i += 1
    shutil.move(src_path, dest_path)
    return dest_path

# --- Model ve scaler'ı yükle ---
best_model = load_model(MODEL_PATH)
if not os.path.exists(SCALER_PATH):
    raise SystemExit(f"Scaler bulunamadı: {SCALER_PATH} (Eğitimde kaydettiğinden emin ol)")
scaler = joblib.load(SCALER_PATH)
print(f"Scaler yüklendi: {SCALER_PATH}")

# --- CSV'yi oku ve normalize et ---
df_all = pd.read_csv(CSV_PATH)
df_all['model_type'] = df_all['model_type'].astype(str).str.strip()
df_all['source_id'] = df_all['source_id'].astype(str).str.strip()
# tabular sütunları sayısala çevir (NaN olanlar sonra elenecek)
for c in TABULAR_COLUMNS:
    df_all[c] = pd.to_numeric(df_all[c], errors='coerce')

# --- Sadece TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1 olan source_id'ler ---
allowed_ids = set(df_all.loc[df_all["model_type"] == "TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1", "source_id"])

# --- Kaynak PNG'lerden yalnızca TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1 olanları filtrele ---
all_pngs = glob.glob(os.path.join(SRC_DIR, "*.png"))
twogauss_pngs = [p for p in all_pngs if base_no_ext(p) in allowed_ids]

if not twogauss_pngs:
    raise SystemExit("Uygun (TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1) PNG bulunamadı.")

random.seed(RANDOM_SEED)
selected_files = random.sample(twogauss_pngs, min(SAMPLE_COUNT, len(twogauss_pngs)))
print(f"TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1 PNG sayısı: {len(twogauss_pngs)} | Seçilen: {len(selected_files)}")

# --- Hedef klasörleri hazırla ---
for cls in CLASSES:
    os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

# --- İndeksle ve işle ---
df = df_all.set_index('source_id')

moved, skipped_nan, skipped_missing, errors = 0, 0, 0, 0

for img_path in selected_files:
    try:
        sid = base_no_ext(img_path)

        if sid not in df.index:
            skipped_missing += 1
            continue

        # (Güvence) TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1 mı?
        if str(df.loc[sid, 'model_type']).strip() != 'TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1':
            continue

        # Tabular veriler
        row = df.loc[sid, TABULAR_COLUMNS]
        # duplicate olasılığına karşı ilk satırı al
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        tab_vals = row.to_numpy(dtype='float32')

        if np.isnan(tab_vals).any():
            skipped_nan += 1
            continue

        tabular_data = scaler.transform(tab_vals.reshape(1, -1))

        # Görsel
        img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Tahmin
        probs = best_model.predict([img_array, tabular_data], verbose=0)[0]
        pred_cls = CLASSES[int(np.argmax(probs))]

        # Taşı (direkt Bedri2/EA|EB|EW)
        dest_dir = os.path.join(TARGET_DIR, pred_cls)
        safe_move(img_path, dest_dir)
        moved += 1
        print(f"{os.path.basename(img_path)} -> {pred_cls}")

    except Exception as e:
        errors += 1
        print(f"Hata: {os.path.basename(img_path)} işlenirken sorun: {e}")

print("===== ÖZET =====")
print(f"Seçilen (TWOGAUSSIANS_WITH_ELLIPSOIDAL_ON_ECLIPSE1) : {len(selected_files)}")
print(f"Taşınan                : {moved}")
print(f"CSV/ID eksik           : {skipped_missing}")
print(f"NaN nedeniyle atlanan  : {skipped_nan}")
print(f"Hata                   : {errors}")
print("İşlem tamamlandı!")
