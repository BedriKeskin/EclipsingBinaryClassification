#!/bin/bash
#SBATCH -p orfoz        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A bkeskin       # Kullanici adi
#SBATCH -J StarShadow        # Gonderilen isin ismi
#SBATCH -o StarShadow.out    # Ciktinin yazilacagi dosya adi
#SBATCH -C weka        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=72:00:00      # Sure siniri koyun.

module load miniconda3
conda activate custom-env
python starshadow.py
