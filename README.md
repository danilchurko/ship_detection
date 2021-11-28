### Airbus Ship Detection Challenge

This repository contains my solution Kaggle's [Airbus Ship Detection challenge](https://www.kaggle.com/c/airbus-ship-detection/code?competitionId=9988&sortBy=scoreDescending&language=Python).

The solution includes several files:

Exploration and analysis of images.

	EDA.ipynb

Run data preparation, augmentation and training of model with training history graphs as output.

	train.py

Run prediction of trained model on first image from test_2 directory.

	inference.py

Config file to easy change parameters.

	conf.py

Native keras implementation of Unet architecture.

	model_unet.py

Function library for decoding masks, generating images for training, etc.

	tools.py

Contains necessary package requirements for solution.

	requirements.txt
