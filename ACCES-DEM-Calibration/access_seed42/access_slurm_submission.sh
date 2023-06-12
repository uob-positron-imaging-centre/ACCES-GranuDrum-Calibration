#!/bin/bash
#SBATCH --time 2:0:0
#SBATCH --mail-type FAIL
#SBATCH --ntasks 2
#SBATCH --nodes 1
#SBATCH --constraint icelake
#SBATCH --wait



        # Commands you'd add in the sbatch script, after `#`
        set -e
        module purge; module load bluebear
        module load bear-apps/2022a
        module load PICI-LIGGGHTS/3.8.1-foss-2022a-VTK-9.2.2
        module load MNE-Python/1.3.1-foss-2022a
        module load coexist/0.3.1-foss-2022a
        module load tqdm/4.64.0-GCCcore-11.3.0
        module load OpenCV/4.6.0-foss-2022a-contrib
        source ${HOME}/virtual-environments/my-virtual-env-${BB_CPU}/bin/activate
    

# Run a single function evaluation with all command-line arguments redirected to Python
python $*
