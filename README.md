# DI-GSNCP-Radar-Sensing
This GitHub repository contains the scripts for the IEEE ICASSP 2024 paper submission: "Multi-Sensor Multi-Scan Multiple Extended Object Radar Sensing".

Author:	Martin Voigt Vejling

E-Mails:	mvv@{math,es}.aau.dk, martin.vejling@gmail.com

In this work a doubly inhomogeneous generalized shot noise Cox process model is proposed to model the measurement process in multi-sensor multi-scan multiple extended object radar sensing scenarios. An estimation procedure based on jump Markov chain Monte Carlo is provided to do target state estimation. For comparison, a spatial proximity baseline is provided in the DBSCAN algorithm with hyperparameters optimized by the posterior of the aforementioned model. Moreover, a theoretically optimal oracle method is also given for comparison.

## Dependencies
This project is created with `Python 3.11.4`

Dependencies:
```
matplotlib 3.7.2
numpy 1.25.2
scipy 1.11.1
scikit-learn 1.2.2
toml 0.10.2
tqdm 4.66.1
line_profiler 4.0.3
tikzplotlib 0.10.1
multiprocess 0.70.15
```

## Usage
To use this GitHub repository follow these steps:

1) Install Python with the dependencies stated in the Dependencies section.
2) Choose numerical experiment settings in sensing_filter.toml.
3) Run sensing_filter_main.py.
