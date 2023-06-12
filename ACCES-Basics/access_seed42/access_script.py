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
# Modifying the "minimal_simulation_script.py" to let ACCES optimise the free
# parameters and minimise the error:
#   1. Declare free parameters at the top, given them names and ranges.
#   2. Use the free parameter values to run the simulation like before.

# ACCES PARAMETERS START

# Unpickle `parameters` from this script's first command-line argument and set
# `access_id` to a unique simulation ID

import coexist
with open(sys.argv[1], 'rb') as f:
    parameters = pickle.load(f)

access_id = int(sys.argv[1].split(".")[-2])
# ACCES PARAMETERS END

# Let's print the `parameters` object
print("ACCES parameters declared:\n", parameters, "\n")

# Extract the parameter values. The rest of the file is the same!
cor = parameters["value"]["CoR"]
ced = parameters["value"]["CED"]
epsilon = parameters["value"]["Epsilon"]
mu = parameters["value"]["Mu"]

# ... here we would launch a simulation, e.g. a GranuDrum simulation
# ... then gather results and compute some objective, e.g. simulated dynamic angle of repose
# ... finally, compare objective to target value, e.g. measured dynamic angle of repose

# For simplicity, here's an analytical 4D Himmelblau function with 8 global
# and 2 local minima - the initial guess is very close to the local one!
error = (cor**2 + ced - 11)**2 + (cor + ced**2 - 7)**2 + epsilon * mu
print(f"For these parameters, the error is {error}")# ACCESS INJECT USER CODE END   ###############################################
###############################################################################


# Save the user-defined `error` and `extra` variables to disk.
with open(sys.argv[2], "wb") as f:
    pickle.dump(error, f)

if "extra" in locals() or "extra" in globals():
    path = os.path.split(sys.argv[2])
    path = os.path.join(path[0], path[1].replace("result", "extra"))
    with open(path, "wb") as f:
        pickle.dump(extra, f)
