import argparse
import json
import logging
import numpy as np
import os
import cyclic_esn
import autoencoder as ae

#.. Initialize logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

#.. Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to data directory", type=str)
parser.add_argument("esnconfig", help="path to ESN config file", type=str)
parser.add_argument("result", help="path to result directory", type=str)
args = parser.parse_args()

#.. Read config file
# config = json.load(open(args.esnconfig + '.json', 'r'))
print('Loading ESN config file.')
print()
config = json.load(open(args.esnconfig, 'r'))

#__ Get training data and test data from specified file
def get_train_test_data(filepath):
    #.. Load time series data
    X, Y = cyclic_esn.load_from_text(filepath)
    #.. Construct training/test sets
    Xtr, Ytr, _, _, Xte, Yte = cyclic_esn.generate_datasets(X, Y)

    return Xtr, Ytr, Xte, Yte

#__ Get output parameter for our ESN
def get_esn_output_parameter(ul, vl, n_ts, tsname):
    predicted_time_series_dic = {}
    sum_errors_dic = {}
    readout_weights_dic = {}

    print('Training ESN.')
    print()
    #.. Train ESN for all triples (u, v, i_time_series)
    for u in ul:
        for v in vl:
            sum = 0.0
            for i_time_series in range(n_ts):
                #.. Get train and test data for our time series data
                file_path = args.data + '/' + tsname \
                            + '_' + str(i_time_series) + '.dat'
                X_tr, Y_tr, X_te, Y_te = get_train_test_data(file_path)

                #.. Train ESN for a set of u, v, and time series data
                #.. and get results for the prediction
                Yhat, error, weight = \
                    cyclic_esn.run_from_config(X_tr, Y_tr, X_te, Y_te,
                                               config, u, v)

                #.. Add result for the dictionary 
                predicted_time_series_dic[(u,v,i_time_series)] = Yhat
                sum += error
                readout_weights_dic[(u,v,i_time_series)] = weight

            #.. sum of all time series data errors for each pair (u,v)
            sum_errors_dic[(u,v)] = sum

    print('Getting optimal (u,v) pair.')
    #.. Get optimal pair (u,v)
    min_u = 0.0
    min_v = 0.0
    (min_u, min_v) = min(sum_errors_dic, key=sum_errors_dic.get)
    print('Optimal (u,v) = ({},{})'.format(min_u, min_v))
    print()

    #.. Prepare output parameters of ESN
    shape_ts = predicted_time_series_dic[(min_u, min_v, 0)].shape
    shape_wv = readout_weights_dic[(min_u, min_v, 0)].shape

    predicted_ts_array = np.empty((shape_ts[0],0), float)
    weights_array = np.empty((shape_wv[0],0), float)

    for i_time_series in range(n_ts):
        predicted_ts_array = \
            np.append(predicted_ts_array,
                      predicted_time_series_dic[(min_u, min_v, i_time_series)], axis=1)
        weight_vector = np.atleast_2d(readout_weights_dic[(min_u, min_v, i_time_series)]).T
        weights_array = np.append(weights_array, weight_vector, axis=1)

    return predicted_ts_array, weights_array

#__ Write ESN results
def output_esn_results(pred_ts_file, pred_ts, weights_file, weights):
    print('Writing predicted time series data and readout weights data.')
    print()
    np.savetxt(pred_ts_file, pred_ts, comments='#')
    np.savetxt(weights_file, weights, comments='#')

#__ Plot weights for 2-dimensional case
def plot_weights(weights):
    #.. 1-dimensional plot
    if weights.shape[0] < 2:
        import matplotlib.pyplot as plt

        for i in range(weights.shape[1]):
            plt.plot(weights[0,i], 0.0, 'o')
            plt.annotate(str(i), xy=(weights[0,i], 0.0))

        plt.title('1-dim readout weight vectors plot')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.grid(True)
        plt.show()

    #.. 2-dimensional plot
    elif weights.shape[0] == 2:
        import matplotlib.pyplot as plt

        for i in range(weights.shape[1]):
            plt.plot(weights[0,i], weights[1,i], 'o')
            plt.annotate(str(i), xy=(weights[0,i], weights[1,i]))

        plt.title('2-dim readout weight vectors plot')
        plt.xlabel('x axis')
        plt.ylabel('y axis')
        plt.grid(True)
        plt.show()

    #.. 3-dimensional plot
    else:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure()
        ax = Axes3D(fig)
        for i in range(weights.shape[1]):
            x = weights[0,i]
            y = weights[1,i]
            z = weights[2,i]
            ax.text(x, y, z, str(i))

        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1.5, 1.5)
        plt.show()

#__ Main function of this script
def main():

    #___ Fit Echo State Network ___#

    print('---- Start calculating ESN model ----')
    print()

    #.. Input parameters of ESN
    ulist = config['ulist']
    vlist = config['vlist']
    n_time_series = config['n_time_series']
    time_series_name = config['time_series_name']

    #.. Train ESN and get output parameters
    predicted_time_series, readout_weights = \
        get_esn_output_parameter(ulist, vlist,
                                 n_time_series, time_series_name)

    #.. output results for ESN
    pred_file = args.result + '/predicted_' + time_series_name + '.dat'
    weights_file = args.result + '/weights_' + time_series_name + '.dat'
    output_esn_results(pred_file, predicted_time_series,
                       weights_file, readout_weights)


    #___ Dimension Reduction by Autoencoder ___#

    #.. Input parameters of Autoencoder
    dim_latent_space = config['dim_latent_space']
    reduction_method = config['reduction_method']
    if reduction_method == 'standard_AE':
        print('---- Dimension reduction by standard autoencoder ----')
        print()
        reduced_weights = ae.standard_AE(readout_weights, dim_latent_space)
        plot_weights(reduced_weights.T)
    elif reduction_method == 'without_reduction':
        plot_weights(readout_weights)
    else:
        print('Dimension reduction method is not specified:')
        print(reduction_method)

    print()
    print('The calculation is successfully completed.')

if __name__ == "__main__":
    main()
