#!/bin/bash

#SBATCH --account=bkeskin
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=72:00:00
#SBATCH --job-name=SStess

#SBATCH --partition=orfoz
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=55
#SBATCH -C weka

###SBATCH --mal-user=bedri.keskin@gmail.com
###SBATCH --mail-type=BEGIN,END,FAIL
###SBATCH --mail-type=ALL

### Load modules
module load miniconda3
conda activate virtualenv

### jobs
python runMultiple.py

exit