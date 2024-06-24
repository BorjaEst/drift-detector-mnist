# drift-detector-mnist

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/drift-detector-mnist/main)](https://jenkins.services.ai4os.eu/job/AI4OS-hub/job/drift-detector-mnist/job/main/)

Drift Detector for MNIST images with Frouros. To launch it, first install the
package then run [deepaas](https://github.com/ai4os/DEEPaaS). See the details
below.

## Installation

> ![warning](https://img.shields.io/badge/Warning-red.svg) **Warning**: If
> you are using a virtual environment, make sure you are working with the last
> version of pip before installing the package.
> Use `pip install --upgrade pip` to upgrade pip.

```bash
git clone https://github.com/ai4os-hub/drift-detector-mnist
cd drift-detector-mnist
pip install -e .
```

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── Dockerfile              <- Steps to build a DEEPaaS API Docker image
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── detector
│   ├── __init__.py         <- Main package containing detection code and models
│   ├── models              <- Subpackage containing models (autoencoder) classes
│   ├── config.py           <- Module with configuration parameters
│   ├── make_autoencoder.py <- Script to create the autoencoder model
│   ├── make_dataset.py     <- Script to create the dataset for building the models
│   ├── make_detector.py    <- Script to create the detector model
│   └── utils.py            <- Module with utilities such loss functions, etc.
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data folder to store the data for building the models
├── docs                    <- A default Sphinx project; see sphinx-doc.org for details
├── models                  <- Folder to store detectors and autoencoders models
├── notebooks               <- Jupyter notebooks.
├── references              <- Data dictionaries, manuals, and all other explanatory materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
├── pyproject.toml          <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│
└── tox.ini                 <- tox file with settings for running tox; see tox.testrun.org
```

## Download and generate datasets

To download and generate the datasets, run the following commands:

```bash
python -m detector.make_dataset
```

It will download the MNIST dataset and generate the processed data required
to train the autoencoder, the drift detector and a model. See the `--help`
option for more information.

This script will generate the following files in the `data` folder
(or the folder specified with the `config.DATA_PATH` option):

```
data
├── MNIST                   <- Folder with the MNIST raw dataset
├── autoencoder_dataset.pt  <- Dataset for training the autoencoder
├── model_dataset.pt        <- Dataset for training the model
└── reference_dataset.pt    <- Dataset for training the drift detector
```

## Train the autoencoder

To optimize the process of detection, we use an autoencoder to reduce the
dimensionality of the input data. To generate the autoencoder, run the following
command:

```bash
python -m detector.make_autoencoder
```

It will generate the autoencoder model and save it in the `models` folder
(or the folder specified with the `config.MODELS_PATH` option) under the
default name `mnist_autoencoder.pt`. See the `--help` option for more
information.

## Train the drift detector

Once the autoencoder is generated, we can train the drift detector. To do so,
run the following command:

```bash
python -m detector.make_detector
```

By default, it will use the autoencoder model generated in the previous step
and generate the drift detector model. The model will be saved in the `models`
folder (or the folder specified with the `config.MODELS_PATH` option) under
default name `mnist_detector.pt`. See the `--help` option for more information.

# Deployment with DEEPaaS

Finally, to deploy the drift detector with DEEPaaS, run the following command:

```bash
deepaas-run --listen-ip 0.0.0.0
```

This will start the DEEPaaS server with the drift detector model. The server
will be available at `http://localhost:5000/ui`. You can configure the server
options by creating a configuration file and passing it as an argument to the
`deepaas-run` command. See the `--help` option and `./deepaas.conf.sample` for
and example configuration file.

## Testing

Running the tests with tox:

```bash
$ pip install -r requirements-dev.txt
$ tox
```

Running the tests with pytest:

```bash
$ pip install -r requirements-test.txt
$ python -m pytest --numprocesses=auto --dist=loadscope tests
```
