# Robo-Eye: Stereo Vision for Autonomous Driving

## Project Overview

**Robo-Eye** is a perception-based project that tackles critical challenges in computer vision for autonomous systems, including:

- **3D Object Detection and Tracking**: Tracking vehicles, pedestrians, and cyclists even under occlusions.
- **Object Classification**: Accurate identification of objects as pedestrians, cyclists, or cars using machine learning techniques.
- **Stereo Camera Calibration**: Ensuring accurate depth perception through camera rectification and calibration.

## Problem Statement

Autonomous driving systems must reliably perceive and interpret their surroundings to navigate safely. Urban scenarios are particularly challenging due to the presence of multiple dynamic entities, such as vehicles and pedestrians, as well as frequent occlusions in the environment. 

The project aims to address the following issues:
1. **Occlusion Management**: Maintaining robust 3D tracking of objects despite partial or complete occlusions.
2. **Accurate Object Classification**: Training models to generalize well on unseen data for precise object recognition.
3. **Stereo Camera Calibration**: Aligning camera views to ensure accurate depth estimation and minimizing calibration errors.

## Features

1. **Object Detection and Tracking**:
   - Detection and 3D tracking of objects in rectified stereo images.
   - Robust handling of occlusions to maintain continuous tracking.

2. **Machine Learning Classification**:
   - A model trained on both 2D and 3D data to classify objects into one of three categories: pedestrians, cyclists, and cars.
   - Use of a custom dataset augmented with web-sourced and self-captured images.

3. **Stereo Camera Calibration**:
   - Calibration and rectification of stereo cameras using provided patterns.
   - Comparison of custom rectification results with pre-rectified data.

## Goals and Milestones

- **3D Tracking and Detection**: Achieve robust tracking of objects in rectified stereo images.
- **Object Classification**: Train a machine learning model and validate it using provided sequences. Test it on unseen data.
- **Stereo Camera Rectification**: Perform stereo camera calibration and compare results to the pre-rectified data.
