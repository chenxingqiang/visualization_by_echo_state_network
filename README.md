# Visualization by echo state network
This is an implementation of visualization for time series
by echo state network using [[1]] as a reference.
The echo state network architecture in [[1]] is
"Simple Cycle Reservoir" in [[2]].
Our code implementation of echo state network is based on [[3]].
In [[1]], the visualization is implemented by combining
echo state network and dimensionality reduction method.
We implemented visualization by the following 2 ways.

1. Only echo state network
2. Echo state netwrok and standard autoencoder

The first method "Only echo state network" means that
the data projection of a time series is implemented by
the readout weight vector itself.
The second method is mentioned in [[1]].
We do not implement ESN-coupled autoencoder,
but only standard autoencoder.

## Requirements
* numpy==1.15.4
* matplotlib==3.0.0
* scipy==1.1.0
* sklearn==0.20.1
* tensorflow==1.12.0
* Keras==2.2.4

## Usage
First, we generate sample data sets.
For the sine data case, we run the python script
in the directory data/sine_data.
```console
$ python sine.py
```
For the disturbed sine data, we generate the data in the same way.
We write the path of data directory, config file, and result directory
in the following way on the shell script file "run.sh".

```console
DATADIR=data/sine_data
CONFIGFILE=configs/sine_config/sine_2dim_without_reduction.json
RESULTDIR=result
```
To start our visualization program, excute the "run.sh" script.

```console
$ ./run.sh
```

## Config file contents
The config file is described in .json file format.
We describe the following items in the config file.

## Datasets
This repository contains the following sample data.

1. Sine function value data in data/sine_data/
2. Disturbed sine function value data in data/disturbed_sine_data/

The first data set contains sine function values changing the amplitude
between [0.1, 0.2,..., 1.0] and the starting angle between
[0.25&pi;, 0.5&pi;,..., 2.0&pi;]. If you run the python script
in data/sine_data/, 80 files will be generated. For each file,
the first column is the input data and the second column is
the target data for our echo state network.
The second data set contains disturbed data of the first ones.

## Results for the two data sets


## References
[[1]] N. Gianniotis, S. D. Kugler, P. Tino, K. L. Polsterer,
"Model-coupled autoencoder for time series visualisation".
 Neurocomputing, vol. 192, pp. 139-146, Jun 2016.

[[2]] A. Rodan, P. Tino,  "Minimum complexity echo state network".
IEEE T. Neural Netw. 22, 131–144 (2011).

[[3]] Python ESN. https://github.com/siloekse/PythonESN


[1]: https://www.sciencedirect.com/science/article/pii/S0925231216002587
[2]: https://ieeexplore.ieee.org/document/5629375
[3]: https://github.com/siloekse/PythonESN