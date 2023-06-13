#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : postprocess.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.06.2023


import shutil
import subprocess
from glob import glob

import numpy as np
import konigcell as kc

import cv2
import pyvista as pv

# ACCES PARAMETERS START
import coexist

parameters = coexist.create_parameters(
    ["sliding", "rolling", "num_particles"],
    minimums = [0.0, 0.0, 20_000],
    maximums = [1.0, 1.0, 40_000],
    values = [0.5, 0.5, 30_000],    # Optional, initial guess / current trial
)
access_id = 0                       # The trial ID
# ACCES PARAMETERS END


# Extract parameter values - ACCES will modify the `parameters` above
sliding = parameters["value"]["sliding"]
rolling = parameters["value"]["rolling"]
num_particles = int(parameters["value"]["num_particles"])


def generate_trial_simulation(trial_directory, rpm):
    '''Generate a trial simulation for the given RPM
    '''

    # Copy the GranuDrum simulation template to its own trial
    shutil.copytree("Template", trial_directory, dirs_exist_ok = True)

    # Read the GranuDrum LIGGGHTS script commands into a list of strings, one
    # for each line of text
    with open(trial_directory + "/granudrum.sim") as f:
        script = f.readlines()

    # Change the lines defining the trial parameters to our modified values
    script[12] = f"variable RotationPeriod equal 60/{rpm}\n"
    script[13] = f"variable NumParticles equal {num_particles}\n"
    script[22] = f"variable SlidingPP equal {sliding}\n"
    script[23] = f"variable SlidingPW equal {sliding}\n"
    script[25] = f"variable RollingPP equal {rolling}\n"
    script[26] = f"variable RollingPW equal {rolling}\n"

    # Save the modified script back into the trial subdirectory
    with open(trial_directory + "/granudrum.sim", "w") as f:
        f.writelines(script)


# 1. Generate two simulations for each trial - one at 15 RPM and one at 45 RPM
trial_15rpm_directory = f"TrialSimulations/sim{access_id:0>4}_15rpm"
generate_trial_simulation(trial_15rpm_directory, rpm = 15)

trial_45rpm_directory = f"TrialSimulations/sim{access_id:0>4}_45rpm"
generate_trial_simulation(trial_45rpm_directory, rpm = 45)


# 2. Launch the two simulations, each within their own directory
proc_15rpm = subprocess.Popen(
    ["liggghts", "-in", "granudrum.sim"],
    cwd = trial_15rpm_directory,
)

proc_45rpm = subprocess.Popen(
    ["liggghts", "-in", "granudrum.sim"],
    cwd = trial_45rpm_directory,
)

# Wait for simulations
proc_15rpm.wait()
proc_45rpm.wait()


# 3. Post-process results


class GranuDrum:
    '''GranuTools GranuDrum system dimensions
    '''
    xlim = [-0.042, +0.042]
    ylim = [-0.042, +0.042]
    radius = 0.042


def encode_u8(image):
    '''Convert image from doubles to uint8 - i.e. encode real values to
    the [0-255] range.
    '''

    u8min = np.iinfo(np.uint8).min
    u8max = np.iinfo(np.uint8).max

    img_min = float(image.min())
    img_max = float(image.max())

    img_bw = (image - img_min) / (img_max - img_min) * (u8max - u8min) + u8min
    img_bw = np.array(img_bw, dtype = np.uint8)

    return img_bw


def image_threshold(image_path, trim = 0.7):
    '''Return the raw and post-processed GranuDrum image in the
    `konigcell.Pixels` format.
    '''

    # Load the image in grayscale and ensure correct orientation:
    #    - x is downwards
    #    - y is rightwards
    image = 255 - cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)[::-1].T

    # Pixellise / discretise the GranuDrum circular outline
    xgrid = np.linspace(GranuDrum.xlim[0], GranuDrum.xlim[1], image.shape[0])
    ygrid = np.linspace(GranuDrum.ylim[0], GranuDrum.ylim[1], image.shape[1])

    # Physical coordinates of all pixels
    xx, yy = np.meshgrid(xgrid, ygrid)

    # Remove the GranuDrum's circular outline
    image[xx ** 2 + yy ** 2 > trim * GranuDrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((7, 7), np.uint8)
    image2 = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # Global thresholding + binarisation
    _, image2 = cv2.threshold(image2, 30, 255, cv2.THRESH_BINARY)
    image2[xx ** 2 + yy ** 2 > trim * GranuDrum.radius ** 2] = 0

    image = kc.Pixels(image, xlim = GranuDrum.xlim, ylim = GranuDrum.ylim)
    image2 = kc.Pixels(image2, xlim = GranuDrum.xlim, ylim = GranuDrum.ylim)

    # Return the original and binarised images
    return image, image2


def simulation_threshold(
    radii,
    positions,
    trim = 0.7,
    image_shape = (512, 512),
):
    '''Return the raw and post-processed occupancy grid of the GranuDrum DEM
    simulation in the `konigcell.Pixels` format.
    '''

    # Concatenate particle trajectories, each separated by a row of NaNs
    num_timesteps = positions.shape[0]
    num_particles = positions.shape[1]

    positions = np.swapaxes(positions, 0, 1)    # (T, P, XYZ) -> (P, T, XYZ)
    positions = np.concatenate(positions)
    positions = np.insert(positions, np.s_[::num_timesteps], np.nan, axis = 0)

    radii = np.swapaxes(radii, 0, 1)            # (T, P) -> (P, T)
    radii = np.concatenate(radii)
    radii = np.insert(radii, np.s_[::num_timesteps], np.nan)

    # Compute residence distribution in the XZ plane (i.e. granular drum side)
    sim_rtd = kc.dynamic2d(
        positions[:, [0, 2]],
        kc.RATIO,
        radii = radii,
        resolution = image_shape,
        xlim = GranuDrum.xlim,
        ylim = GranuDrum.ylim,
        verbose = False,
    )

    # Post-process the NumPy array of pixels within `sim_rtd`
    sim_pix = sim_rtd.pixels

    # Pixellise / discretise the GranuDrum circular outline
    xgrid = np.linspace(GranuDrum.xlim[0], GranuDrum.xlim[1], image_shape[0])
    ygrid = np.linspace(GranuDrum.ylim[0], GranuDrum.ylim[1], image_shape[1])

    # Physical coordinates of all pixels
    xx, yy = np.meshgrid(xgrid, ygrid)

    # Remove the GranuDrum's circular outline
    sim_pix[xx ** 2 + yy ** 2 > trim * GranuDrum.radius ** 2] = 0

    # Inflate then deflate the uint8-encoded image
    kernel = np.ones((7, 7), np.uint8)
    sim_pix2 = cv2.morphologyEx(
        encode_u8(sim_pix),
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Global thresholding + binarisation
    _, sim_pix2 = cv2.threshold(sim_pix2, 10, 255, cv2.THRESH_BINARY)
    sim_pix2[xx ** 2 + yy ** 2 > trim * GranuDrum.radius ** 2] = 0

    sim_rtd2 = kc.Pixels(
        sim_pix2,
        xlim = GranuDrum.xlim,
        ylim = GranuDrum.ylim,
    )

    # Colour GranuDrum's background in the raw pixellated image
    sim_pix[
        (xx ** 2 + yy ** 2 < trim * GranuDrum.radius ** 2) &
        (sim_pix2 == 0.)
    ] = 7 / 255 * sim_pix.max()

    # Return the original and binarised images
    return sim_rtd, sim_rtd2


def extract_timestep(filename):
    '''Extract timestep number from a VTK filename, for example
    "parent/directory/particles_142000.vtk" -> 142000
    '''
    ending = filename.split("_")[-1]        # -> "142000.vtk"
    timestep = ending.split(".")[0]         # -> "142000"
    return int(timestep)


def compute_error(trial_directory, rpm = 15):
    '''Compute error for simulation in trial_directory, for a given RPM
    '''

    # Discover all VTK files saved by the LIGGGHTS simulation
    files = [
        f for f in glob(f"{trial_directory}/post/particles_*.vtk")
        if "boundingBox" not in f
    ]
    files = sorted(files, key=extract_timestep)

    if len(files) == 0:
        raise FileNotFoundError("No VTK files found - did the simulation run?")

    # Get range of particle IDs in the simulation
    particles = pv.read(files[0])
    idmin = particles["id"].min()
    idmax = particles["id"].max()

    # Pre-allocate empty arrays of positions and radii
    num_timesteps = len(files)
    num_particles = idmax - idmin + 1
    positions = np.full((num_timesteps, num_particles, 3), np.nan)
    radii = np.full((num_timesteps, num_particles), np.nan)

    # Read in VTK files and save particle data sorted by IDs
    for i in range(num_timesteps):
        particles = pv.read(files[i])
        positions[i, particles["id"] - idmin, :] = particles.points
        radii[i, particles["id"] - idmin] = particles["radius"]

    # Post-process image and simulation and compute error between them
    image_path = f"Measurements/gd_nature_{rpm}rpm_averaged.bmp"

    trim = 0.6
    img_raw, img_post = image_threshold(image_path, trim = trim)
    sim_raw, sim_post = simulation_threshold(radii, positions, trim = trim,
                                             image_shape = img_raw.pixels.shape)

    # Pixel physical dimensions, in mm
    dx = (GranuDrum.xlim[1] - GranuDrum.xlim[0]) / img_post.pixels.shape[0] * 1000
    dy = (GranuDrum.ylim[1] - GranuDrum.ylim[0]) / img_post.pixels.shape[1] * 1000

    # The error is the total different area, i.e. the number of pixels with
    # different values times the area of a pixel
    error = np.sum(img_post.pixels != sim_post.pixels) * dx * dy
    print(f"For RPM = {rpm}, error to experiment = {error} mm2")

    return error


error_15rpm = compute_error(trial_15rpm_directory, rpm = 15)
error_45rpm = compute_error(trial_45rpm_directory, rpm = 45)


# 4. Set error variable to contain both RPM errors
error = [error_15rpm, error_45rpm]
