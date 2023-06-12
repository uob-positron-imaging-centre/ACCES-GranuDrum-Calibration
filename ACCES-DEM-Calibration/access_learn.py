#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : access_learn.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.06.2023


# Use ACCES to learn a simulation's parameters
import coexist

# Initialise ACCES with the script running a single trial and let it work its magic
access = coexist.Access("run_simulation_trial.py")
access.learn(
    num_solutions = 8,          # Number of solutions per epoch
    target_sigma = 0.1,         # Target std-dev (accuracy) of solution
    random_seed = 42,           # Reproducible / deterministic optimisation
)
