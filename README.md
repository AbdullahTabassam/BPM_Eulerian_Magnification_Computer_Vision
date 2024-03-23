# Project Name: Eulerian Magnification for Heart Rate Measurement

## Overview
This project implements Eulerian Magnification to amplify color variations in video frames, particularly emphasizing subtle changes in skin color caused by blood flow. Leveraging computer vision techniques, it aims to measure heart rate non-invasively by analyzing these color changes in facial regions.

## Motivation
Accurate and non-intrusive heart rate measurement is crucial in various fields, including healthcare, sports science, and human-computer interaction. Traditional methods often require physical contact or specialized equipment, whereas Eulerian Magnification offers a promising approach by analyzing video data.

## Implementation Details
The core of this project is based on the research paper titled ["Eulerian Video Magnification for Revealing Subtle Changes in the World"](https://people.csail.mit.edu/mrub/papers/vidmag.pdf). I adapted the methods outlined in the paper, employing Python along with OpenCV and MediaPipe libraries for efficient processing and analysis of video frames.

Key components of our implementation include:
- Face recognition using MediaPipe for region of interest selection.
- Utilization of Gaussian and Laplacian pyramids for frequency-based filtering.
- Applying Eulerian magnification techniques to amplify subtle color variations.
- Heart rate estimation through frequency analysis of color changes in the facial region.

## Dependencies
- Python 3.x
- OpenCV
- MediaPipe
- Additional package used: CVZone
---

*Disclaimer: This project is for educational and research purposes only. It should not be used as a substitute for professional medical advice or diagnosis.*
