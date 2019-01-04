# Visualization by echo state network
This is an implementation of visualization for time series
by echo state network using [[1]] as a reference.
The echo state network architecture in [[1]] is
"Simple Cycle Reservoir" in [[2]].
Our code implementation of echo state network is based on [[3]].
In [[1]], the visualization is implemented by combining
echo state network and dimension reduction method.
We implemented visualization by the following 2 ways.

1. Echo state network without dimension reduction
2. Echo state netwrok with standard autoencoder

The first method "Echo state network without dimension reduction"
means that the data projection of a time series is implemented by
the readout weight vector itself.
The second method is mentioned in [[1]].
"ESN" means "Echo State Network" in this repository.
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
First, we generate sample datasets.
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
The directory configs/ contains config files.
Config files are described in .json file format.
We describe the following items in a config file.

|Name of the item       |Meaning of the item                                   |
|:----                  |:----	                                               |
|n_internal_units       |number of internal units for ESN                      |
|n_time_series          |number of time series in a dataset                   |
|signs_of_input_weights |signs of the vector components of input weight for ESN|
|ulist 			|list of absolute values of input weights for ESN      |
|vlist 			|list of absolute values of internal weights for ESN   |
|reduction_method 	|name of reduction method    	       		       |
|dim_latent_space 	|number of units of hidden layer for autoencoder       |

### Note
* "n_time_series" means the number of .dat file in a dataset directory.
In the case of sine function value data, "n_time_series" is 80.
* "signs_of_input_weights" is specified by a vector whose all components
have absolute value 1.0.
* Visualization needs a fixed pair (u, v) for a dataset.
 Our program will find the optimal pair (u, v) among "ulist" and "vlist".
* If we use "ESN without dimension reduction" method for the visualization,
set "reduction_method" to "without_reduction".
If we use "ESN with standard autoencoder" method, set "reduction_method"
to "standard_AE".

## Datasets
This repository contains the following sample data.

1. Sine function value data in data/sine_data/
2. Disturbed sine function value data in data/disturbed_sine_data/

The first dataset directory data/sine_data/ contains sine function values
changing the amplitude between [0.1, 0.2,..., 1.0] and the starting
angle between [0.25&pi;, 0.5&pi;,..., 2.0&pi;].
If you run the python script in data/sine_data/, 80 files will be generated.
For each file, the first column is the input data and the second column is
the target data for our echo state network.
The second dataset directory data/disturbed_sine_data/ contains
disturbed data of the first one. "disturbed" means that the data is disturbed
by random numbers generated from normal distribution having mean 0 and variance
0.03.

## Results for the two datasets
We show the results of visualization of the two datasets
for "Echo state network without dimension reduction" method.

### sine function value dataset
1. The dimension of readout weight vector is 1

When the dimension of the readout weight vector is 1,
we have 3 clusters as in the following figure.

![sine_1dim](https://github.com/kazu-riemann/visualization_by_echo_state_network/blob/images/sine_1dim.png)

Each cluster corresponds to some starting angle of
the sine function.

2. The dimension of readout weight vector is 2

When the dimension of the readout weight vector is 2,
we have again 3 clusters as in the following figure.

![sine_2dim](https://github.com/kazu-riemann/visualization_by_echo_state_network/blob/images/sine_2dim_without_reduction.png)

In this case, the clusters seem to lie on lines
that are parallel to each other.

### disturbed sine function value dataset
The following figures are the results for disturbed
sine function value datasets. In both cases, the plots
do not have exact clusters such as in the sine function
value dataset.

1. The dimension of readout weight vector is 1

![dis_sine_1dim](https://github.com/kazu-riemann/visualization_by_echo_state_network/blob/images/dis_sine_1dim.png)

2. The dimension of readout weight vector is 2

![dis_sine_2dim](https://github.com/kazu-riemann/visualization_by_echo_state_network/blob/images/dis_sine_2dim_without_reduction.png)

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