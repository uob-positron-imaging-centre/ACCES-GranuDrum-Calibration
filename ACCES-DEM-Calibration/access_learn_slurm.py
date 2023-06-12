#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.06.2023


# Use ACCES to learn a simulation's parameters
import coexist
from coexist.schedulers import SlurmScheduler


scheduler = SlurmScheduler(
    "8:0:0",           # Time allocated for a single simulation
    ntasks = 2,
    nodes = 1,
    constraint = "icelake",
    commands = '''
        # Commands you'd add in the sbatch script, after `#`
        set -e
        module purge; module load bluebear
        module load bear-apps/2022a
        module load PICI-LIGGGHTS/3.8.1-foss-2022a-VTK-9.2.2
        module load MNE-Python/1.3.1-foss-2022a
        module load coexist/0.3.1-foss-2022a
        module load tqdm/4.64.0-GCCcore-11.3.0
        source ${HOME}/virtual-environments/my-virtual-env-${BB_CPU}/bin/activate
    ''',
)


# Initialise ACCES with the script running a single trial and let it work its magic
access = coexist.Access("run_simulation_trial.py", scheduler)
access.learn(
    num_solutions = 8,          # Number of solutions per epoch
    target_sigma = 0.1,         # Target std-dev (accuracy) of solution
    random_seed = 42,           # Reproducible / deterministic optimisation
)
