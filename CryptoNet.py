from model import build_LSTM_model
import time
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import statistics


class CryptoNet:

    def __init__(self, data_file, layers, sequence_length, deep):
        self.data = data_file
        self.layers = layers
        self.seq_len = sequence_length
        self.model = None
        self.train_data = None   # [input_train, output_train]
        self.test_data = None    # [prediction_data, test_data]
        self.deep = deep



    def prepare_data(self, test=False):
        """Prepare the data by reading in one sequence, spliting into multiple,
           scale the data for Keras layers, and then split into test/train data
           TODO: Add cross validation split of data"""

        f = open(self.data, 'rb').read()
        data = f.decode().split('\n')

        # Split the data into sequences for LSTM model
        temp_seq_len = self.seq_len + 1
        sequences = []
        for num in range(len(data) - temp_seq_len):
            seq = data[num: num + temp_seq_len]
            sequences.append(seq)

        # Normalize the data so that the percent change in the sequence is represented.
        # e.g. [0.0, -0.05577030433238028, -0.15701283981526903, -0.17998578992776548, .... -0.015487287060375832]
        scaled_data = []
        for sequence in sequences:
            scaled_sequence = [((float(p) / float(sequence[0])) - 1) for p in sequence]
            scaled_data.append(scaled_sequence)

        # Put data into numpy array and section off data into needed parts for model
        scaled_data = np.array(scaled_data)

        # split into train and test sections
        num_training_rows = round(.9 * scaled_data.shape[0])
        training_data = scaled_data[:int(num_training_rows), :]
        np.random.shuffle(training_data)


        # Training data
        input_train = training_data[:, :-1]  # input training data
        output_train = training_data[:, -1]  # output target data

        # reshape input data
        input_train = np.reshape(input_train, (input_train.shape[0],
                                               input_train.shape[1], 1))
        # return training data
        self.train_data = [input_train, output_train]


        # Testing data
        prediction_input = scaled_data[int(num_training_rows):, :-1]  # input prediction data
        true_data = scaled_data[int(num_training_rows):, -1]  # correct data that we will plot against

        # Reshape prediction input data
        prediction_input = np.reshape(prediction_input, (prediction_input.shape[0],
                                                         prediction_input.shape[1], 1))
        # return testing data
        self.test_data = [prediction_input, true_data]



    def train_model(self, batch, training_cycles, optimizer, valid_split):

        # Start the clock
        training_time = time.time()

        # Build the model to be trained
        self.model = build_LSTM_model(self.layers, optimizer, self.deep)

        # Assert prepare_data has been called
        assert self.train_data is not None
        input_train = self.train_data[0]
        output_train = self.train_data[1]

        # Fit the model to parameters
        self.model.fit(input_train, output_train,
                       batch_size=batch, nb_epoch=training_cycles,
                       validation_split=valid_split)

        print('Training lasted ->  ', time.time() - training_time)


    def predict_trends(self, trend_length):
        """Predict a number of trends with the length given as a parameter"""
        final_trends = []
        predict_data = self.test_data[0]
        num_trends = int(len(predict_data) / trend_length)

        for x in range(num_trends):
            window = predict_data[x * trend_length]
            trend_predictions = []
            for y in range(trend_length):
                current_trend = self.model.predict(window[newaxis, :, :])[0, 0]
                trend_predictions.append(current_trend)

                # Decrease the size of the window
                window = window[1:]

                # Insert prediction into window
                window = np.insert(window, [self.seq_len-1], trend_predictions[-1], axis=0)

            final_trends.append(trend_predictions)
        return final_trends


    def evaluate_performance(self, predictions, trend_length):
        """Return the average trend error of each of the predictions based on the
           factual data provided in test_data"""
        trend_errors = [0] * len(predictions)
        correct_data = self.test_data[1]
        i = 0
        for predict in predictions:
            for trend in predict:
                trend_errors[int(i/trend_length)] += abs(trend - correct_data[i])
                i += 1

        average_trend_error = statistics.mean(trend_errors)
        return average_trend_error


    def plot_trends(self, trend_predictions, trend_length, error):
        """Plots the various trends predicted by the model against a factually
           correct set of data."""

        true_data = self.test_data[1]

        graph = plt.figure()
        true_data_line = graph.add_subplot(111)
        true_data_line.plot(true_data, label='Historical Data')

        # Add padding for the plotted graph
        for x, datum in enumerate(trend_predictions):
            padding = [None for y in range(x * trend_length)]
            plt.plot(padding + datum)
        plt.ylabel("Percent change in BTC/USD")
        plt.xlabel("Data from Predictions")
        plt.text(1, 1, "Trend Length: " + str(trend_length))
        plt.text(.9, .9, "Error: " + str(error))
        plt.legend()
        plt.show()


if __name__ == '__main__':

    deep = True
    if deep:

        # Test Deep Neural Network

        DeepCN = CryptoNet(data_file='data/BTC_Dateless.csv', sequence_length=75, layers=[1, 75, 300, 900, 1], deep=True)
        DeepCN.prepare_data(test=True)
        DeepCN.train_model(batch=400, training_cycles=10, optimizer='rmsprop', valid_split=0.1)

        metrics = []
        trend_length_intervals = [5, 10, 15, 20, 30]
        for interval in trend_length_intervals:
            trends = DeepCN.predict_trends(trend_length=interval)
            average_error = DeepCN.evaluate_performance(trends, interval)
            DeepCN.plot_trends(trends, interval, average_error)
            metrics += (interval, average_error)

        print(metrics)

    else:

        # Test Neural Network

        BasicCN = CryptoNet(data_file='data/BTC_Dateless.csv', sequence_length=50, layers=[1, 50, 100, 1], deep=False)
        BasicCN.prepare_data(test=True)
        BasicCN.train_model(batch=400, training_cycles=10, optimizer='rmsprop', valid_split=0.1)

        metrics = []
        trend_length_intervals = [5, 10, 15, 20, 30]
        for interval in trend_length_intervals:
            trends = BasicCN.predict_trends(trend_length=interval)
            average_error = BasicCN.evaluate_performance(trends, interval)
            BasicCN.plot_trends(trends, interval, average_error)
            metrics += (interval, average_error)

        print(metrics)





