import numpy as np
from astropy.timeseries import LombScargle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Örnek: Çok sayıda ışık eğrisi verisi (time ve flux sütunları)
# Bu her bir ışık eğrisi verisi için farklı dosyalar olabilir, onları döngüyle işleyeceğiz.
lightcurves = ['lightcurve1.txt', 'lightcurve2.txt', 'lightcurve3.txt', ...]  # Burada dosya isimlerini girin
labels = [0, 1, 1, 0, ...]  # Sınıflandırma için etiketler, her bir ışık eğrisinin sınıfı (örneğin binary system türü)

strongest_frequencies = []

# Her ışık eğrisi için en güçlü frekansı bul ve listeye ekle
for lightcurve in lightcurves:
    # Işık eğrisi dosyasını yükle
    data = np.loadtxt(lightcurve)
    time = data[:, 0]
    flux = data[:, 1]

    # Lomb-Scargle periodogram uygula
    frequencies, power = LombScargle(time, flux).autopower()

    # En güçlü frekansı bul
    max_power_index = np.argmax(power)
    strongest_frequency = frequencies[max_power_index]

    # En güçlü frekansı listeye ekle
    strongest_frequencies.append(strongest_frequency)

# Girdi verileri (X) ve etiketler (y) hazırlanıyor
X = np.array(strongest_frequencies).reshape(-1, 1)  # Girdi verileri (strongest_frequency'ler)
y = np.array(labels)  # Etiketler

# Veriyi eğitim ve test setine böl (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Makine öğrenmesi modelini oluştur ve eğit
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Modeli test et
y_pred = model.predict(X_test)

# Modelin doğruluğunu ölç
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

# Test setindeki örneklerin tahminleri
for i, (true_label, pred_label) in enumerate(zip(y_test, y_pred)):
    print(f"Gerçek etiket: {true_label}, Tahmin edilen etiket: {pred_label}")
