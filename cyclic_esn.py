import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def NRMSE(y_true, y_pred):
    """ Normalized Root Mean Squared Error """
    y_std = np.std(y_true)

    return np.sqrt(mean_squared_error(y_true, y_pred))/y_std

class ESN(object):
    def __init__(self, n_internal_units = 100,
                 connectivity = 0.5, input_scaling = 0.5, input_shift = 0.0,
                 teacher_scaling = 0.5, teacher_shift = 0.0,
                 noise_level = 0.01, u = 0.9, v = 0.5, sign_vector = None):
        # Initialize attributes
        self._n_internal_units = n_internal_units
        self._connectivity = connectivity

        self._input_scaling = input_scaling
        self._input_shift = input_shift
        self._teacher_scaling = teacher_scaling
        self._teacher_shift = teacher_shift
        self._noise_level = noise_level
        self._dim_output = None

        # The weights will be set later, when data is provided
        self._input_weights = None
        self._internal_weights = None

        # Regression method and embedding method.
        # Initialized to None for now. Will be set during 'fit'.
        self._regression_method = None
        self._embedding_method = None

    #__ fit the model without return value
    def fit(self, Xtr, Ytr, n_drop = 0, regression_method = 'linear',
            regression_parameters = None, embedding = 'identity', n_dim = 3,
            embedding_parameters = None, u = 0.9, v = 0.5, sign_vector = None):

        _,_ = self._fit_transform(Xtr = Xtr, Ytr = Ytr, n_drop = n_drop,
                                  regression_method = regression_method,
                                  regression_parameters = regression_parameters,
                                  embedding = embedding, n_dim = n_dim,
                                  embedding_parameters = embedding_parameters,
                                  u = u, v = v, sign_vector = sign_vector)

        return

    #__ fit the model with states as return value
    def _fit_transform(self, Xtr, Ytr, n_drop = 0, regression_method = 'linear',
                       regression_parameters = None, embedding = 'identity',
                       n_dim = 3, embedding_parameters = None,
                       u = 0.9, v = 0.5, sign_vector = None):

        n_data, dim_data = Xtr.shape
        _, dim_output = Ytr.shape

        self._dim_output = dim_output

        #.. If this is the first time the network is tuned,
        #.. set the input and internal weights.
        if (self._input_weights is None):
            self._initialize_input_weights(u, sign_vector)

        if (self._internal_weights is None):
            self._initialize_internal_weights(v)

        # Initialize regression method
        # Ridge regression
        self._regression_method = Ridge(alpha = regression_parameters)

        # Initialize embedding method
        self._embedding_dimensions = self._n_internal_units

        # Calculate states/embedded states.
        # Note: If the embedding is 'identity', embedded states will be equal to the states.
        states, embedded_states,_ = self._compute_state_matrix(X = Xtr, Y = Ytr, n_drop = n_drop)

        # Train output
        # self._regression_method.fit(np.concatenate((embedded_states, self._scaleshift(Xtr[n_drop:,:], self._input_scaling, self._input_shift)), axis=1), self._scaleshift(Ytr[n_drop:,:], self._teacher_scaling, self._teacher_shift).flatten())

        self._regression_method.fit(states,
                                    self._scaleshift(Ytr[n_drop:,:], self._teacher_scaling,
                                                     self._teacher_shift).flatten())
        return states, embedded_states

    def predict(self, X, Y = None, n_drop = 0, error_function = NRMSE):
        Yhat, error, weights, _, _ = \
        self._predict_transform(X = X, Y = Y, n_drop = n_drop,
                                error_function = error_function)

        return Yhat, error, weights

    def _predict_transform(self, X, Y = None, n_drop = 0, error_function = NRMSE):
        # Predict outputs
        states,embedded_states,Yhat = self._compute_state_matrix(X = X, n_drop = n_drop)
        weights = self._regression_method.coef_
        # print('readout_weights:')
        # print(weights)

        # Revert scale and shift
        Yhat = self._uscaleshift(Yhat, self._teacher_scaling, self._teacher_shift)

        # Compute error if ground truth is provided
        if (Y is not None):
            error = error_function(Y[n_drop:,:], Yhat)

        return Yhat, error, weights, states, embedded_states

    def _compute_state_matrix(self, X, Y = None, n_drop = 0):
        n_data, _ = X.shape

        # Initial values
        previous_state = np.zeros((1, self._n_internal_units), dtype=float)
        previous_output = np.zeros((1, self._dim_output), dtype=float)

        # Storage
        state_matrix = \
            np.empty((n_data - n_drop, self._n_internal_units), dtype=float)
        embedded_states = \
            np.empty((n_data - n_drop, self._embedding_dimensions), dtype=float)
        outputs = np.empty((n_data - n_drop, self._dim_output), dtype=float)

        for i in range(n_data):
            # Process inputs
            previous_state = np.atleast_2d(previous_state)
            current_input = np.atleast_2d(self._scaleshift(X[i, :], self._input_scaling, self._input_shift))

            # Calculate state. Add noise and apply nonlinearity.
            state_before_tanh = self._internal_weights.dot(previous_state.T) \
                + self._input_weights.dot(current_input.T)
            state_before_tanh += \
                np.random.rand(self._n_internal_units, 1)*self._noise_level
            previous_state = np.tanh(state_before_tanh).T

            # Embed data and perform regression if applicable.
            if (Y is not None):
                # If we are training, the previous output should be a scaled and shifted version of the ground truth.
                previous_output = self._scaleshift(Y[i, :], self._teacher_scaling, self._teacher_shift)
            else:
                # Should the data be embedded?
                if (self._embedding_method is not None):
                    current_embedding = self._embedding_method.transform(previous_state)
                else:
                    current_embedding = previous_state

                # Perform regression
                # previous_output = self._regression_method.predict(np.concatenate((current_embedding, current_input), axis=1))
                previous_output = self._regression_method.predict(current_embedding)

            # Store everything after the dropout period
            if (i > n_drop - 1):
                state_matrix[i - n_drop, :] = previous_state.flatten()

                # Only save embedding for test data.
                # In training, we do it after computing the whole state matrix.
                if (Y is None):
                    embedded_states[i - n_drop, :] = current_embedding.flatten()

                outputs[i - n_drop, :] = previous_output.flatten()

        # Now, embed the data if we are in training
        if (Y is not None):
            if (self._embedding_method is not None):
                embedded_states = self._embedding_method.fit_transform(state_matrix)
            else:
                embedded_states = state_matrix

        return state_matrix, embedded_states, outputs

    def _scaleshift(self, x, scale, shift):
        # Scales and shifts x by scale and shift
        return (x*scale + shift)

    def _uscaleshift(self, x, scale, shift):
        # Reverts the scale and shift applied by _scaleshift
        return ( (x - shift)/float(scale) )

    def _initialize_input_weights(self, absolute_value, sign_vector):

        #.. Choose a set of sign for input weights randomly
        # binary_array = np.random.binomial(n=1, p=0.5, size=n_internal_units)
        # input_weights = np.asarray([])
        # for bin in binary_array:
        #     if bin > 0.5:
        #         input_weights = np.append(input_weights, 1.0)
        #     else:
        #         input_weights = np.append(input_weights, -1.0)

        #.. Set an absolute value of the components of input weights
        sign_vector = np.array(sign_vector)
        self._input_weights = absolute_value * sign_vector
        self._input_weights = np.atleast_2d(self._input_weights).T
        # print('Initialize input weights:')
        # print(self._input_weights)

    def _initialize_internal_weights(self, value):
        # Generate a cyclic arrangement matrix
        internal_weights = np.zeros((self._n_internal_units, self._n_internal_units))
        for i in range(self._n_internal_units -1):
            internal_weights[i+1, i] = value
        internal_weights[0, self._n_internal_units -1] = value
        self._internal_weights = internal_weights
        # print('Initialize internal weights:')
        # print(internal_weights)


def run_from_config(Xtr, Ytr, Xte, Yte, config, u, v):
    #.. Instantiate ESN object
    esn = ESN(n_internal_units = config['n_internal_units'],
              connectivity = config['connectivity'],
              input_scaling = config['input_scaling'],
              input_shift = config['input_shift'],
              teacher_scaling = config['teacher_scaling'],
              teacher_shift = config['teacher_shift'],
              noise_level = config['noise_level'],
              u = u, v = v, sign_vector = config['signs_of_input_weights'])

    #.. Get parameters
    n_drop = config['n_drop']
    regression_method = config['regression_method']
    regression_parameters = config['regression_parameters']
    embedding = config['embedding']
    n_dim = config['n_dim']
    embedding_parameters = config['embedding_parameters']

    #.. Fit our network
    esn.fit(Xtr, Ytr, n_drop = n_drop, regression_method = regression_method,
            regression_parameters = regression_parameters,
            embedding = embedding, n_dim = n_dim,
            embedding_parameters = embedding_parameters,
            u = u, v = v, sign_vector = config['signs_of_input_weights'])

    Yhat, error, weights = esn.predict(Xte, Yte)
    return Yhat, error, weights


def format_config(n_internal_units, connectivity,
                  input_scaling, input_shift, teacher_scaling, teacher_shift,
                  noise_level, n_drop, regression_method, regression_parameters,
                  embedding, n_dim, embedding_parameters):

    config = dict(
                n_internal_units = n_internal_units,
                connectivity = connectivity,
                input_scaling = input_scaling,
                input_shift = input_shift,
                teacher_scaling = teacher_scaling,
                teacher_shift = teacher_shift,
                noise_level = noise_level,
                n_drop = n_drop,
                regression_method = regression_method,
                regression_parameters = regression_parameters,
                embedding = embedding,
                n_dim = n_dim,
                embedding_parameters = embedding_parameters
            )

    return config

def generate_datasets(X, Y, test_percent = 0.15, val_percent = 0.15, scaler = StandardScaler):
    n_data,_ = X.shape

    n_te = np.ceil(test_percent*n_data).astype(int)
    n_val = np.ceil(val_percent*n_data).astype(int)
    n_tr = n_data - n_te - n_val

    # Split dataset
    Xtr = X[:n_tr, :]
    Ytr = Y[:n_tr, :]

    Xval = X[n_tr:-n_te, :]
    Yval = Y[n_tr:-n_te, :]

    Xte = X[-n_te:, :]
    Yte = Y[-n_te:, :]

    # Scale
    Xscaler = scaler()
    Yscaler = scaler()

    # Fit scaler on training set
    Xtr = Xscaler.fit_transform(Xtr)
    Ytr = Yscaler.fit_transform(Ytr)

    # Transform the rest
    Xval = Xscaler.transform(Xval)
    Yval = Yscaler.transform(Yval)

    Xte = Xscaler.transform(Xte)
    Yte = Yscaler.transform(Yte)

    return Xtr, Ytr, Xval, Yval, Xte, Yte

def construct_output(X, shift):
    return X[:-shift,:], X[shift:, :]

def load_from_text(path):
    data = np.loadtxt(path, comments='#')

    return np.atleast_2d(data[:, 0]).T, np.atleast_2d(data[:, 1]).T

if __name__ == "__main__":
    pass
