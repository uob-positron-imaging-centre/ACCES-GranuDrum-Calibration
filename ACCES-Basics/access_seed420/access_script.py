#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : async_access_template.py
# License: GNU v3.0


'''Run a user-defined simulation script with a given set of free parameter
values, then save the `error` value to disk.

ACCES takes an arbitrary simulation script that defines its set of free
parameters between two `# ACCESS PARAMETERS START / END` directives and
substitutes them with an ACCESS-predicted solution. After the simulation, it
saves the `error` variable to disk.

This simulation setup is achieved via a form of metaprogramming: the user's
code is modified to change the `parameters` to what is predicted at each run,
then code is injected to save the `error` variable. This generated script is
called in a massively parallel environment with two command-line arguments:

    1. The path to this run's `parameters`, as predicted by ACCESS.
    2. A path to save the user-defined `error` variable to.

You can find them in the `access_seed<seed>/results` directory.
'''


import os
import sys
import pickle


###############################################################################
# ACCESS INJECT USER CODE START ###############################################
# ACCESS PARAMETERS START

# Unpickle `parameters` from this script's first command-line argument and set
# `access_id` to a unique simulation ID

import coexist
import numpy as np
with open(sys.argv[1], 'rb') as f:
    parameters = pickle.load(f)

access_id = int(sys.argv[1].split(".")[-2])
# ACCESS PARAMETERS END


x, y = parameters["value"]
print(parameters)


# Multi-objective optimisation problem taken from:
# http://www.cs.uccs.edu/~jkalita/work/cs571/2012/MultiObjectiveOptimization.pdf
a1 = 0.5 * np.sin(1) - 2 * np.cos(1) + np.sin(2) - 1.5 * np.cos(2)
a2 = 1.5 * np.sin(1) - np.cos(1) + 2 * np.sin(2) - 0.5 * np.cos(2)

b1 = 0.5 * np.sin(x) - 2 * np.cos(x) + np.sin(y) - 1.5 * np.cos(y)
b2 = 1.5 * np.sin(x) - np.cos(x) + 2 * np.sin(y) - 0.5 * np.cos(y)

f1 = 1 + (a1 - b1)**2 + (a1 - b2)**2
f2 = (x + 3)**2 + (y + 1)**2

error = [-f1, -f2]
print(f"For these parameters, the errors are {error}")# ACCESS INJECT USER CODE END   ###############################################
###############################################################################


# Save the user-defined `error` and `extra` variables to disk.
with open(sys.argv[2], "wb") as f:
    pickle.dump(error, f)

if "extra" in locals() or "extra" in globals():
    path = os.path.split(sys.argv[2])
    path = os.path.join(path[0], path[1].replace("result", "extra"))
    with open(path, "wb") as f:
        pickle.dump(extra, f)
