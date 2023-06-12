# Example script running a "simulation" for some given parameters
cor = 0.5
ced = 1.2
epsilon = 1.5
mu = 4.2

# Print the set parameter values
print("CoR / CED / Epsilon / Mu:", cor, ced, epsilon, mu)

# ... here we would launch a simulation, e.g. a GranuDrum simulation
# ... then gather results and compute some objective, e.g. simulated dynamic angle of repose
# ... finally, compare objective to target value, e.g. measured dynamic angle of repose

# For simplicity, here's an analytical 4D Himmelblau function with 8 global
# and 2 local minima - the initial guess is very close to the local one!
error = (cor**2 + ced - 11)**2 + (cor + ced**2 - 7)**2 + epsilon * mu
print(f"For these parameters, the error is {error}")