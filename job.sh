#!/bin/bash
#SBATCH -p akya-cuda        # Kuyruk adi: Uzerinde GPU olan kuyruk olmasina dikkat edin.
#SBATCH -A bkeskin       # Kullanici adi
#SBATCH -J FourierFit        # Gonderilen isin ismi
#SBATCH -o FourierFit.out    # Ciktinin yazilacagi dosya adi
#SBATCH --gres=gpu:0        # Her bir sunucuda kac GPU istiyorsunuz? Kumeleri kontrol edin.
#SBATCH -N 1                # Gorev kac node'da calisacak?
#SBATCH -n 1                # Ayni gorevden kac adet calistirilacak?
#SBATCH --cpus-per-task 10  # Her bir gorev kac cekirdek kullanacak? Kumeleri kontrol edin.
#SBATCH --time=149:00:00      # Sure siniri koyun.

eval "$(/truba/home/$USER/miniconda3/bin/conda shell.bash hook)"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/truba/home/bkeskin/miniconda3/envs/dl-env/lib/
conda activate dl-env
python FourierFit.py
