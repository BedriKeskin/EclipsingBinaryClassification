#!/bin/bash

#SBATCH --account=bkeskin
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=72:00:00
#SBATCH --job-name=Keplerstarshadow

#SBATCH --partition=orfoz
#SBATCH --ntasks=10
#SBATCH --nodes=10
#SBATCH --cpus-per-task=110
#SBATCH -C weka

###SBATCH --mal-user=bedri.keskin@gmail.com
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --mail-type=ALL

### Load modules
module load miniconda3
conda activate virtualenv

### jobs
python starshadow.py

exit