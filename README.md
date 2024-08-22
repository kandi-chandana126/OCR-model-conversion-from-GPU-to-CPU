This repository contains an Optical Character Recognition (OCR) application that was originally designed to run on a GPU. This README will guide you through the steps to convert the OCR application to run on a CPU instead of a GPU.
Introduction
This project utilizes the EasyOCR library to perform OCR on video frames. Initially, the application leverages GPU acceleration for faster processing. However, there are cases where running the application on a CPU is more desirable or necessary, such as when a GPU is unavailable or when testing on low-end hardware.

Requirements
Python 3.7 or higher
EasyOCR (pip install easyocr)
OpenCV (pip install opencv-python)
psutil (pip install psutil)
editdistance (pip install editdistance)
Matplotlib (pip install matplotlib)
