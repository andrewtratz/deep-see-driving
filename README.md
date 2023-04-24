 Deep See Driving
====================

## Overview

This repository contains code for a graduate student project for Harvard Extension School's
DGMD E-17 Robotics, Autonomous Vehicles, Drones, and Artificial Intelligence class.

This project builds a data capture platform consisting of a stereo camera pair
and Velodyne Puck Hi-Res LiDAR sensor. Using the captured data, it trains a computer vision
model to predict depth maps, relying solely on the stereo camera
sensor data as an input.

## Robot setup on StereoPi v2 with Raspberry Pi Compute Module 4

* Download OS image from https://www.mediafire.com/file/bf7wwnvb6vv9tgb/opencv-4.5.3.56.zip/file
* Install OS image on SD card using Raspberry Pi installer
    Settings:
        - Hostname: testpi.local
        - Enable SSH
        - user: pi
        - password: deepsee
        - Wireless LAN config
* ssh into testpi.local
* sudo raspi-config
    - set max resolution
    - enable VNC
    - enable camera
    - expand filesystem
    - enable predictable network interface names
    - reboot
* sudo apt update
* sudo apt install git
* git clone https://github.com/andrewtratz/deep-see-driving
* sudo apt-get install tcpdump
* pip install --pre scapy[basic]

# Capturing data on the Robot

To capture stereo camera data on the robot, run the capture.py
script from the ./Robot subfolder.

While camera data is being recorded, the LiDAR data should be dumped
to .pcap files using the tcpdump command line utility.

# Calibrating the sensors

Default sensor calibration settings are specified in **calibration_settings.py**.

If calibration adjustments are required, the file **manual_calibration.py** can be
used to overlay LiDAR and camera images and test out modifications to the
predefined calibration values.

# Preprocessing

Two scripts are provided for preprocessing the camera images and LiDAR data, respectively.
To use, first update the data paths at the beginning of the scripts:
* process_raw_images.py
* process_raw_pcap.py

# Training

To train a model, update the path of the preprocessed data files in **train.py**
and run this program.

# Inference

The python program **inference.py** can take a single image as input (update path
in the file), run inference given a set of pretrained model weights (also indicated
in the file), and produce a visual output of the results.