# CS 6998 - Final Project - Fundamentals of Speech Recognition

### Author: Jared Samet
### UNI: jss2272

This repo contains the project code for the prosody feature extraction for my final project for CS 6998, Fundamentals of Speech Recognition.

### Final Paper

The [Final Paper](latex/final-report.pdf) is included here.

### REQUIREMENTS

```
sudo apt-get install python-tk
sudo apt-get install ffmpeg
sudo pip install -r requirements.txt
```

### USAGE

```
./annotate_only.sh <path-to-wav> <path-to-text>
```

or

```
./transcribe_and_annotate.sh <path-to-wav>
```

### This portion of the project is written using Python 2, for compatibility with Kaldi.

The speech synthesis portion was implemented using a lightly-modified fork of Keith Ito's Tacotron implementation. Although the trained model is included in this repo under the [trained_model](trained_model) folder, that implementation is written in Python 3 and uses Tensorflow. In order to synthesize audio as well as create the cluster labels, you will need to install a python environment manager (I'm using pyenv) so that the two systems don't interfere with one another. Providing step-by-step instructions for setting that environment up, along with all of the requirements to install Tensorflow/cuDNN/CUDA/etc., is beyond the scope of this project, but I would be delighted to help if you are interested.

My fork of Tacotron can be found [here](https://github.com/oracleofnj/tacotron). Once your separate environment and Tensorflow are set up properly, installation is relatively straightforward.
