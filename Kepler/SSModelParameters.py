import os
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def method_name():
    #global f, reader, row, period
    with open(os.path.join(root, file), newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row and row[0].strip() == "id":
                Id = row[1].strip()
            if row and row[0].strip() == "stage":
                stage = row[1].strip()
            if row and row[0].strip() == "t_mean":
                t_mean = row[1].strip()
            if row and row[0].strip() == "period":
                period = row[1].strip()
            if row and row[0].strip() == "t_1":
                t_1 = row[1].strip()
            if row and row[0].strip() == "t_2":
                t_2 = row[1].strip()
            if row and row[0].strip() == "t_1_1":
                t_1_1 = row[1].strip()
            if row and row[0].strip() == "t_1_2":
                t_1_2 = row[1].strip()
            if row and row[0].strip() == "t_2_1":
                t_2_1 = row[1].strip()
            if row and row[0].strip() == "t_2_2":
                t_2_2 = row[1].strip()
            if row and row[0].strip() == "t_b_1_1":
                t_b_1_1 = row[1].strip()
            if row and row[0].strip() == "t_b_1_2":
                t_b_1_2 = row[1].strip()
            if row and row[0].strip() == "t_b_2_1":
                t_b_2_1 = row[1].strip()
            if row and row[0].strip() == "t_b_2_2":
                t_b_2_2 = row[1].strip()
            if row and row[0].strip() == "depth_1":
                depth_1 = row[1].strip()
            if row and row[0].strip() == "depth_2":
                depth_2 = row[1].strip()
            if row and row[0].strip() == "ecosw_form":
                ecosw_form = row[1].strip()
            if row and row[0].strip() == "esinw_form":
                esinw_form = row[1].strip()
            if row and row[0].strip() == "cosi_form":
                cosi_form = row[1].strip()
            if row and row[0].strip() == "phi_0_form":
                phi_0_form = row[1].strip()
            if row and row[0].strip() == "log_rr_form":
                log_rr_form = row[1].strip()
            if row and row[0].strip() == "log_sb_form":
                log_sb_form = row[1].strip()
            if row and row[0].strip() == "e_form":
                e_form = row[1].strip()
            if row and row[0].strip() == "w_form":
                w_form = row[1].strip()
            if row and row[0].strip() == "i_form":
                i_form = row[1].strip()
            if row and row[0].strip() == "r_sum_form":
                r_sum_form = row[1].strip()
            if row and row[0].strip() == "r_rat_form":
                r_rat_form = row[1].strip()
            if row and row[0].strip() == "sb_rat_form":
                sb_rat_form = row[1].strip()
            if row and row[0].strip() == "ecosw_phys":
                ecosw_phys = row[1].strip()
            if row and row[0].strip() == "esinw_phys":
                esinw_phys = row[1].strip()
            if row and row[0].strip() == "cosi_phys":
                cosi_phys = row[1].strip()
            if row and row[0].strip() == "phi_0_phys":
                phi_0_phys = row[1].strip()
            if row and row[0].strip() == "log_rr_phys":
                log_rr_phys = row[1].strip()
            if row and row[0].strip() == "log_sb_phys":
                log_sb_phys = row[1].strip()
            if row and row[0].strip() == "e_phys":
                e_phys = row[1].strip()
            if row and row[0].strip() == "w_phys":
                w_phys = row[1].strip()
            if row and row[0].strip() == "i_phys":
                i_phys = row[1].strip()
            if row and row[0].strip() == "r_sum_phys":
                r_sum_phys = row[1].strip()
            if row and row[0].strip() == "r_rat_phys":
                r_rat_phys = row[1].strip()
            if row and row[0].strip() == "sb_rat_phys":
                sb_rat_phys = row[1].strip()
            if row and row[0].strip() == "ratio_1_1":
                ratio_1_1 = row[1].strip()
            if row and row[0].strip() == "ratio_1_2":
                ratio_1_2 = row[1].strip()
            if row and row[0].strip() == "ratio_2_1":
                ratio_2_1 = row[1].strip()
            if row and row[0].strip() == "ratio_2_2":
                ratio_2_2 = row[1].strip()
            if row and row[0].strip() == "ratio_3_1":
                ratio_3_1 = row[1].strip()
            if row and row[0].strip() == "ratio_3_2":
                ratio_3_2 = row[1].strip()
            if row and row[0].strip() == "ratio_4_1":
                ratio_4_1 = row[1].strip()
            if row and row[0].strip() == "ratio_4_2":
                ratio_4_2 = row[1].strip()
    return Id, stage, t_mean, period, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2, ecosw_form, esinw_form, cosi_form, phi_0_form, log_rr_form, log_sb_form, e_form, w_form, i_form, r_sum_form, r_rat_form, sb_rat_form, ecosw_phys, esinw_phys, cosi_phys, phi_0_phys, log_rr_phys, log_sb_phys, e_phys, w_phys, i_phys, r_sum_phys, r_rat_phys, sb_rat_phys, ratio_1_1, ratio_1_2, ratio_2_1, ratio_2_2, ratio_3_1, ratio_3_2, ratio_4_1, ratio_4_2

X = []

for root, _, files in os.walk("StarShadowAnalysis"):
    for file in files:
        if file.endswith("_summary.csv"):
            #print(os.path.join(root, file))

            Id, stage, t_mean, period, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2, ecosw_form, esinw_form, cosi_form, phi_0_form, log_rr_form, log_sb_form, e_form, w_form, i_form, r_sum_form, r_rat_form, sb_rat_form, ecosw_phys, esinw_phys, cosi_phys, phi_0_phys, log_rr_phys, log_sb_phys, e_phys, w_phys, i_phys, r_sum_phys, r_rat_phys, sb_rat_phys, ratio_1_1, ratio_1_2, ratio_2_1, ratio_2_2, ratio_3_1, ratio_3_2, ratio_4_1, ratio_4_2 = method_name()
            #print(Id, stage, t_mean, period, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2)

            if stage == "9":
                X.append([period, t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2, ecosw_form, esinw_form, cosi_form, phi_0_form, log_rr_form, log_sb_form, e_form, w_form, i_form, r_sum_form, r_rat_form, sb_rat_form, ecosw_phys, esinw_phys, cosi_phys, phi_0_phys, log_rr_phys, log_sb_phys, e_phys, w_phys, i_phys, r_sum_phys, r_rat_phys, sb_rat_phys, ratio_1_1, ratio_1_2, ratio_2_1, ratio_2_2, ratio_3_1, ratio_3_2, ratio_4_1, ratio_4_2])

X = np.array([[float(val) for val in row] for row in X])

'''
# Küme sayısını belirlemek için Elbow Method ve Silhouette Score hesapla
inertia = []
silhouette_scores = []
cluster_range = range(2, 10)

for k in cluster_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Elbow Method Grafiği
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cluster_range, inertia, marker='o')
plt.xlabel("Küme Sayısı")
plt.ylabel("Inertia (Küme içi hata)")
plt.title("Elbow Method - En İyi Küme Sayısı")

# Silhouette Score Grafiği
plt.subplot(1, 2, 2)
plt.plot(cluster_range, silhouette_scores, marker='o', color='red')
plt.xlabel("Küme Sayısı")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - Küme Kalitesi")

plt.show()
4 çıktı
'''

# K-Means ile kümeleme (4 küme)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)

# GMM ile kümeleme (4 küme)
gmm = GaussianMixture(n_components=4, random_state=42, n_init=10)
labels_gmm = gmm.fit_predict(X)

# PCA ile 2D'ye indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Görselleştirme
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# K-Means Grafiği
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='viridis', edgecolors='k')
axes[0].set_title("K-Means Kümeleme (4 Küme)")
axes[0].set_xlabel("PCA Bileşeni 1")
axes[0].set_ylabel("PCA Bileşeni 2")
fig.colorbar(scatter1, ax=axes[0], label="Cluster ID")

# GMM Grafiği
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_gmm, cmap='plasma', edgecolors='k')
axes[1].set_title("GMM Kümeleme (4 Küme)")
axes[1].set_xlabel("PCA Bileşeni 1")
axes[1].set_ylabel("PCA Bileşeni 2")
fig.colorbar(scatter2, ax=axes[1], label="Cluster ID")

plt.show()

# Veriyi ölçeklendir (DBSCAN ölçek duyarlıdır)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN kümeleme
dbscan = DBSCAN(eps=1.5, min_samples=5)  # eps ve min_samples ayarlarını optimize etmek gerekebilir
labels_dbscan = dbscan.fit_predict(X_scaled)

# PCA ile 2D'ye indirgeme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DBSCAN Sonuçlarını Görselleştirme
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_dbscan, cmap='rainbow', edgecolors='k')
plt.colorbar(scatter, label="Cluster ID")
plt.xlabel("PCA Bileşeni 1")
plt.ylabel("PCA Bileşeni 2")

# Kümeleme sonuçlarını özetleyelim
unique_clusters = np.unique(labels_dbscan)
num_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)  # Gürültü (-1) hariç küme sayısı
num_noise_points = list(labels_dbscan).count(-1)  # Gürültü noktalarının sayısı

plt.title("DBSCAN num_clusters: %d" %num_clusters + " total_points: %d" %len(X) + " num_noise_points: %d" %num_noise_points)
plt.show()
