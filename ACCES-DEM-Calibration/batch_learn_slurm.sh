#!/bin/bash
#SBATCH --time 80:00:00
#SBATCH --ntasks 1
#SBATCH --nodes 1
#SBATCH --constraint icelake
#SBATCH --job-name gd-acces
#SBATCH --output access_slurm_%j.out


set -e
module purge; module load bluebear

module load bear-apps/2022a
module load PICI-LIGGGHTS/3.8.1-foss-2022a-VTK-9.2.2
module load MNE-Python/1.3.1-foss-2022a
module load coexist/0.3.1-foss-2022a
module load tqdm/4.64.0-GCCcore-11.3.0
source ${HOME}/virtual-environments/my-virtual-env-${BB_CPU}/bin/activate


# Launch ACCES - it will then launch its own SLURM processes
python access_learn_slurm.py
