<h1 align="center">
i-GRIP: a vision-based interface for grasping intention detection and grip selection
</h1>

<div align="center">
<h3>
<a href="https://github.com/emoullet">Etienne Moullet</a>,
<a href="http://imagine.enpc.fr/~aubrym/">Christine Azevedo-Coste</a>,
<a href="https://github.com/fbailly">François Bailly</a>,
<a href="https://jcarpent.github.io/">Justin Carpentier</a>
</h3>
</div>

<!-- # TODO -->
<!-- - Add the script for visualization. -->
<!-- - Upload the BOP zip files to gdrive. -->

# Table of content

- [Overview](#overview)
- [Installation](#installation)
- [Downloading and preparing data](#downloading-and-preparing-data)
- [Note on GPU parallelization](#note-on-gpu-parallelization)
- [Reproducing single-view results](#reproducing-single-view-results)
- [Training the single-view 6D pose estimation models](#training-the-single-view-6D-pose-estimation-models)
  - [Synthetic data generation](#synthetic-data-generation)
  - [Training script](#training-script)
- [Reproducing multi-view results](#reproducing-multi-view-results)
- [Using CosyPose in a custom scenario](#using-cosypose-in-a-custom-scenario)
- [BOP20 models and results](#bop20-models-and-results)

# Overview

Grasping is crucial for many daily activities, and its impairment considerably impacts quality of life and autonomy. Attempts to restore this function may rely on various approaches and devices (functional electrical stimulation, exoskeletons, prosthesis…) with command modalities often exert considerable cognitive loads on users and lack controllability and intuitiveness in daily life . i-GRIP paves the way to novel user interface for grasping movement control in which the user delegates the grasping task decisions to the device, only moving their (potentially prosthetic) hand toward the targeted object.

### Algorithmic structure

Required information for assisting an ongoing grasping task is the following: 1) hand position and orientation; 2) object position and orientation; 3) object nature (including shape and potentially weight and texture). We use an OAK-D S2 (Luxonis) stereoscopic RGB camera as a data acquisition sensor. Hand pose estimation is performed using <a href="https://developers.google.com/mediapipe/solutions/vision/hand_landmarker">Google’s MediaPipe</a>, leveraging stereoscopic vision for depth estimation with <a href="https://docs.luxonis.com/en/latest/"> DepthAI </a>. Object identification and pose estimation is achieved using <a href="https://github.com/Simple-Robotics/cosypose">CosyPose</a>, a multi-object 6D pose estimator trained on a set of objects with known 3D models.

Three metrics are concurrently used to analyse a hand's movement :

- distance to every detected objects
- time derivative of the distance to every detected objects
- impacts on every detected objects' meshes of cones of rays builted uppon the extrapolated hand's trajectory

# Installation

```
conda env create -f environment.yml
```

The installation may take some time as several packages must be downloaded and installed/compiled.

As torch install are rarely well handled in environment.yml install, run:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Clone cosypose repository from <a href="https://github.com/Simple-Robotics/cosypose?tab=readme-ov-file#installation">here </a>, without installing dependecies from environment.yml. That is to say :

1. either : - use gitkraken to clone the repo, initiate submodules and pull LFS files - `git clone --recurse-submodules https://github.com/Simple-Robotics/cosypose.git
cd cosypose
git lfs pull`
2. `python setup.py develop`

Note :
in `cosypose/lib3d/transform.py` you might need to comment the 4th line :

```
# eigenpy.switchToNumpyArray()
```

In your i-GRIP folder, run

```
pip install -e .
```

# Downloading and preparing data

Required data paths are declared in `config.py` and may be adapted to your own architecture

## Cosypose

Follow <a href="https://github.com/Simple-Robotics/cosypose?tab=readme-ov-file#downloading-and-preparing-data">these instructions </a> to download neural networks and 3d models.
Depending on the performances of your setup, the ray tracing used in the algorithm may be slow with the original, high definition ycbv and t-less 3d meshes. If that is the case, you may run the script `simplify_meshes.py`, and adjust the constants `_*dataset*_TARGET_COUNT` to your preferences.

## Mediapipe

see : https://developers.google.com/mediapipe/solutions/vision/hand_landmarker

or dl directly :
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Hardware

## Camera

i-GRIP was designed and tested using <a href="https://shop.luxonis.com/collections/oak-cameras-1">OAK-D cameras</a> to recontruct depth maps along rgb frames. Yet, i-GRIP could in theory be used with any rgbd device, at the cost of rewritting your own `RgbdCameras.py` file.

## Object datasets

Cosypose was trained on two different datasets, that can be indistinguishly be used in i-GRIP :

- <a href="https://www.ycbbenchmarks.com/">YCBV</a>
- <a href="http://cmp.felk.cvut.cz/t-less/">T-LESS</a>

If you don't have real objects from these datasets at your disposal, you can downloand sample images from <a href="">TODO</a>

# Using i-GRIP

In your i_grip environment, run :

```
python run_i_grip.py
```
