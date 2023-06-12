#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : postprocess.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.06.2023


import os
import sys
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

# If trial ID is supplied via command-line
if len(sys.argv) > 1:
    access_id = sys.argv[1]
else:
    access_id = 0
# ACCES PARAMETERS END


# Extract parameter values - ACCES will modify the `parameters` above
sliding = parameters.loc["sliding", "value"]
rolling = parameters.loc["rolling", "value"]
num_particles = int(parameters.loc["num_particles", "value"])


# Simulation directories
trial_15rpm_directory = f"TrialSimulations/sim{access_id:0>4}_15rpm"
trial_45rpm_directory = f"TrialSimulations/sim{access_id:0>4}_45rpm"

if not os.path.isdir(trial_15rpm_directory):
    raise FileNotFoundError("Trial simulation at 15 RPM not found!")

if not os.path.isdir(trial_45rpm_directory):
    raise FileNotFoundError("Trial simulation at 45 RPM not found!")


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



def compute_plot_error(trial_directory, rpm = 15):
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


    # Plot experiment, simulation and the two superimposed
    import plotly

    # Create colormap starting from white
    cm = plotly.colors.sequential.Blues
    cm[0] = 'rgb(255,255,255)'

    fig = kc.create_fig(
        ncols = 3,
        subplot_titles = ["Experiment", "Superimposed", "Simulation"],
    )

    # Plot "raw", untrimmed images
    img_raw, img_post = image_threshold(image_path, trim = 1.)
    sim_raw, sim_post = simulation_threshold(radii, positions, trim = 1.,
                                             image_shape = img_raw.pixels.shape)

    # Plot GranuDrum photograph on the left
    fig.add_trace(img_raw.heatmap_trace(colorscale = cm), 1, 1)

    # Plot LIGGGHTS simulation on the right
    fig.add_trace(sim_raw.heatmap_trace(colorscale = cm), 1, 3)

    # Plot both simulation and experiment, colour-coding differences in the middle
    diff = np.zeros(img_raw.pixels.shape)
    diff[(img_post.pixels == 255) & (sim_post.pixels == 255)] = 64
    diff[(img_post.pixels == 255) & (sim_post.pixels == 0)] = 128
    diff[(img_post.pixels == 0) & (sim_post.pixels == 255)] = 192

    diff = kc.Pixels(diff, img_raw.xlim, img_raw.ylim)

    # "Whiten" / blur the areas not used
    xgrid = np.linspace(diff.xlim[0], diff.xlim[1], diff.pixels.shape[0])
    ygrid = np.linspace(diff.ylim[0], diff.ylim[1], diff.pixels.shape[1])
    xx, yy = np.meshgrid(xgrid, ygrid)
    diff.pixels[xx ** 2 + yy ** 2 > trim * GranuDrum.radius ** 2] *= 0.2

    fig.add_trace(diff.heatmap_trace(colorscale = cm), 1, 2)

    # Show plot
    fig.show(width=1200, height=500)

    # Return error
    return error


error_15rpm = compute_plot_error(trial_15rpm_directory, rpm = 15)
error_45rpm = compute_plot_error(trial_45rpm_directory, rpm = 45)
