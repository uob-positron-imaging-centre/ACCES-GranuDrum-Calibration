# Modifying the "minimal_simulation_script.py" to let ACCES optimise the free
# parameters and minimise the error:
#   1. Declare free parameters at the top, given them names and ranges.
#   2. Use the free parameter values to run the simulation like before.

# ACCES PARAMETERS START
import coexist

# Declare the parameters here - give them names and ranges.
# ACCES will modify this section
parameters = coexist.create_parameters(
    ["CoR", "CED", "Epsilon", "Mu"],
    minimums = [-5, -5, -5, -5],
    maximums = [+5, +5, +5, +5],
)
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
print(f"For these parameters, the error is {error}")