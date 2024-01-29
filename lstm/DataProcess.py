import numpy as np
from sklearn import preprocessing


# deal data format the data , inputs are the different columns but the same sequence with same timing
# n_steps_int：the n sequencial inputs ; n_steps_out : n sequencial outputs (which the timing follows the input); offset : the timing offset for output (follows inputs)
def split_sequences(input_sequences, output_sequence, n_steps_in, n_steps_out, offset = 0):
    X, y = list(), list() # instantiate X and y
    for i in range(len(input_sequences)):
        # find the end of the input, output sequence
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the dataset
        if out_end_ix + offset> len(input_sequences): break
        # gather input and output of the pattern
        seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix + offset:out_end_ix + offset]
        # seq_x, seq_y = input_sequences[i:end_ix], output_sequence[end_ix + offset:out_end_ix + offset, -1]
        X.append(seq_x), y.append(seq_y)
    return np.array(X), np.array(y)


def dataNormalization(data, min=0, max=1):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(min, max), copy=False)
    x_minmax = min_max_scaler.fit_transform(data)
    #data_normalized = x_minmax.transform(data)
    return min_max_scaler, x_minmax

def dataReinverse(min_max_scaler, data):
    actual_predictions = min_max_scaler.inverse_transform(data)
    return actual_predictions


