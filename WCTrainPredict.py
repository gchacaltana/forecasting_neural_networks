#!/usr/bin/env python3
"""
Interactive pipeline to train a neural network and forecast monthly
drinking-water consumption for a household.

Loads historical data, builds a supervised time-series matrix, trains a
feed-forward Keras model, and predicts the next month's consumption (m3).
"""

import os

from core.constants import (
    DEFAULT_ACTIVATION,
    DEFAULT_EPOCHS,
    DEFAULT_LOSS,
    DEFAULT_OPTIMIZER,
    DEFAULT_TEST_PERCENTAGE,
    DEFAULT_TRAIN_PERCENTAGE,
    DEFAULT_TRAIN_PREDICTIVE_MONTHS,
    FEATURE_RANGE,
    PLT_FIGURE_SIZE,
    PLT_STYLE,
    PREDICTION_DECIMAL_PLACES,
    TF_CPP_MIN_LOG_LEVEL,
)

# Must be set before importing keras/tensorflow to reduce log noise.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL

import matplotlib.pylab as plt
import pandas as pd
from keras.layers import Dense, Flatten
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from core import DatasetNotFoundError, InvalidPartitionError
from Forecasting.Helpers.Functions import Console
from Forecasting.Models.DataModel.WaterConsumptionDataModel import (
    WaterConsumptionDataModel,
)


class WCTrainPredict:
    """Interactive trainer and forecaster for monthly water consumption."""

    def __init__(self) -> None:
        """Initialize plotting, app state, and neural-network defaults."""
        self.set_config_app()
        self.set_config_neural_network()
        self.set_config_plt()

    def set_config_app(self) -> None:
        self.df = None
        self.wc_normalize = None
        self.df_normalize = None
        self.building = None
        self.apartment = None
        self.predicted_value = None

    def set_config_plt(self) -> None:
        plt.rcParams['figure.figsize'] = PLT_FIGURE_SIZE
        plt.style.use(PLT_STYLE)

    def set_config_neural_network(self) -> None:
        self.epochs = DEFAULT_EPOCHS
        self.train_predictive_months = DEFAULT_TRAIN_PREDICTIVE_MONTHS
        # matrix_sl: supervised-learning matrix for TS (Time Series)
        self.matrix_sl = None
        self.train_percentage = DEFAULT_TRAIN_PERCENTAGE
        self.test_percentage = DEFAULT_TEST_PERCENTAGE
        self.data_train = []
        self.data_test = []
        self.x_train, self.y_train = [], []
        self.x_val, self.y_val = [], []
        self.model_nn = None  # neural network model
        self.activation = DEFAULT_ACTIVATION
        self.optimizer = DEFAULT_OPTIMIZER
        self.loss = DEFAULT_LOSS

    def run(self) -> None:
        """Run the full interactive train-and-predict pipeline."""
        self.input_data()
        self.build_wc_df()
        self.get_time_series_wc()
        self.normalize_wc()
        self.iterate_train()
    
    def iterate_train(self) -> None:
        """
        Iterative method until training is confirmed as finished.
        """
        self.convert_time_series_to_supervised_learning_matrix()
        self.data_partition()
        self.create_model_neural_network()
        self.execution_model_neural_network()
        self.results_model_neural_network()
        self.question_continue_prediction()

    def input_data(self) -> None:
        """
        Collect building and apartment names to load the dataset from the database.
        """
        Console.highlight("1. SELECT DATASET")
        self.building = str(input("Enter building name: ")).upper()
        self.apartment = str(input("Enter apartment name: ")).upper()

    def build_wc_df(self) -> None:
        """
        Build the water-consumption dataframe.
        """
        Console.highlight(f"2. MONTHLY WATER CONSUMPTION FOR APARTMENT {self.apartment}")
        self.get_dataset_water_consumption()
        self.set_dataframe_wc()
        self.show_dataframe_wc()
        Console.stop_continue(f"[View consumption for {self.apartment}]")
        print(self.df)
        Console.stop_continue("[Press Enter to continue]")

    def get_dataset_water_consumption(self) -> None:
        """
        Fetch water-consumption records from the database.
        """
        wcdm = WaterConsumptionDataModel()
        self.wm_consumptions = wcdm.get_wm_month_consumption_by_property(self.building, self.apartment)
        if len(self.wm_consumptions) == 0:
            raise DatasetNotFoundError(
                f"Apartment {self.apartment} in building {self.building} "
                "has no monthly consumption records."
            )
    
    def set_dataframe_wc(self) -> None:
        """
        Create the water-consumption (wc) dataframe.
        """
        self.df = pd.DataFrame(
            self.wm_consumptions,
            columns=['Anio', 'Mes', 'fecha_facturacion', 'm3', 'Facturado'],
        )
        self.df['fecha_facturacion'] = pd.to_datetime(self.df['fecha_facturacion'])
        self.df = self.df.set_index('fecha_facturacion')
    
    def show_dataframe_wc(self) -> None:
        """
        Print dataset summary information.
        """
        print(f"Observations: {len(self.df.index)} monthly water consumption records")
        print(f"Minimum date: {(self.df.index.min()).strftime('%d/%m/%Y')}")
        print(f"Maximum date: {(self.df.index.max()).strftime('%d/%m/%Y')}")

    def get_time_series_wc(self) -> None:
        """
        Build and display the time series used for training.
        """
        self.df = self.df.drop(['Anio', 'Mes', 'Facturado'], axis=1)
        print("TIME SERIES")
        print(self.df)
        Console.stop_continue("[Press Enter to continue]")

    def normalize_wc(self) -> None:
        """
        Normalize m3 consumption values to the range [-1, 1].
        """
        Console.highlight("3. DATA NORMALIZATION")
        scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
        self.wc_normalize = scaler.fit_transform(self.df.values.reshape(-1, 1))
        print(f"Time series data normalized to [{FEATURE_RANGE[0]}, {FEATURE_RANGE[1]}].")
        print(self.wc_normalize)
        Console.stop_continue("[Press Enter to continue]")

    def convert_time_series_to_supervised_learning_matrix(self) -> None:
        """
        Convert the time series into a supervised-learning matrix with as many
        predictor variables as entered via the console.
        """
        Console.highlight("4. CONVERT TIME SERIES TO SUPERVISED LEARNING MATRIX")
        self.train_predictive_months = int(input("Enter the number of predictor variables: "))
        self.matrix_sl = self.convert_sequence_to_matrix(1)
        print(
            f"Time series matrix with {self.train_predictive_months} "
            "predictor variables (training)"
        )
        print("\n")
        print(self.matrix_sl)
        Console.stop_continue("[Press Enter to continue]")

    def convert_sequence_to_matrix(self, number_target_y: int = 1) -> pd.DataFrame:
        """
        Parameters
        ----------
        number_target_y: Number of target variables (estimated y)
        """
        self.number_target_y = number_target_y
        self.df_normalize = pd.DataFrame(self.wc_normalize)
        self.matrix_cols, self.matrix_cols_names = list(), list()
        self.set_predictors_columns_matrix()
        self.set_target_column_matrix()
        matrix = pd.concat(self.matrix_cols, axis=1)
        matrix.columns = self.matrix_cols_names
        # Drop missing values
        matrix.dropna(inplace=True)
        return matrix

    def set_predictors_columns_matrix(self) -> None:
        """
        Add predictor variables to the matrix.
        """
        for i in range(self.train_predictive_months, 0, -1):
            self.matrix_cols.append(self.df_normalize.shift(i))
            self.matrix_cols_names += [f"v_pred(t-{i})"]

    def set_target_column_matrix(self) -> None:
        """
        Add target variables to the matrix.
        """
        n_vars = 1 if isinstance(self.wc_normalize, list) else self.wc_normalize.shape[1]
        for i in range(0, self.number_target_y):
            self.matrix_cols.append(self.df_normalize.shift(-i))
            if i == 0:
                self.matrix_cols_names += [('y(t)')]
            else:
                self.matrix_cols_names += [f"v_y_{j + 1}(t+{i})" for j in range(n_vars)]

    def data_partition(self) -> None:
        Console.highlight("5. DATASET PARTITION")
        self.input_data_partition()
        self.set_data_partition()
        self.build_xy_train_test()
        self.print_data_partition()
        Console.stop_continue("[Press Enter to continue]")

    def input_data_partition(self) -> None:
        """
        Read train and test percentages from the console for the data split.
        """
        self.train_percentage = int(input("Enter percentage for training data: "))
        self.test_percentage = int(input("Enter percentage for test data: "))
        percentage_sum = self.train_percentage + self.test_percentage
        if percentage_sum != 100:
            raise InvalidPartitionError(
                "The sum of the training and test percentages is invalid"
            )

    def set_data_partition(self) -> None:
        values = self.matrix_sl.values
        total_rows = len(values)
        self.n_train = int(round(total_rows * (self.train_percentage / 100), 0))
        self.n_test = total_rows - self.n_train
        self.data_train = values[:self.n_train, :]
        self.data_test = values[self.n_train:, :]

    def build_xy_train_test(self) -> None:
        self.x_train, self.y_train = self.data_train[:, :-1], self.data_train[:, -1]
        self.x_val, self.y_val = self.data_test[:, :-1], self.data_test[:, -1]
        self.x_train = self.x_train.reshape(
            (self.x_train.shape[0], 1, self.x_train.shape[1])
        )
        self.x_val = self.x_val.reshape((self.x_val.shape[0], 1, self.x_val.shape[1]))

    def print_data_partition(self) -> None:
        print("\n")
        print(f"TRAINING ({self.train_percentage} %) : {self.n_train} observations (months)")
        print(f"TEST ({self.test_percentage} %) : {self.n_test} observations (months)")

    def create_model_neural_network(self) -> None:
        """
        Create a feed-forward neural network model.
        Architecture:
        1 hidden layer with "n" neurons.
        Activation function: Hyperbolic Tangent (values in [-1, 1]).
        Optimizer: Adam
        Loss metric: Mean Absolute Error (MAE)
        Accuracy metric: Mean Squared Error (MSE)
        """
        Console.highlight("6. CREATING THE NEURAL NETWORK")
        self.print_info_neural_network()
        self.neurals_hidden_layer = int(input("Neurons for the input layer: "))
        print("\n")
        self.model_nn = Sequential()
        self.model_nn.add(
            Dense(
                self.neurals_hidden_layer,
                input_shape=(1, self.train_predictive_months),
                activation=self.activation,
            )
        )
        self.model_nn.add(Flatten())
        self.model_nn.add(Dense(1, activation=self.activation))
        self.model_nn.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=["mse"]
        )
        self.model_nn.summary()
        Console.stop_continue("[Press Enter to continue]")

    def print_info_neural_network(self) -> None:
        print("+ 1 hidden layer with n neurons")
        print("+ Output: 1 neuron")
        print("+ Activation function: Hyperbolic Tangent")
        print("+ Optimizer: Adam")
        print("+ Loss function: Mean Absolute Error (MAE)")
        print("+ Accuracy metric: Mean Squared Error (MSE)")
        print("\n\n")

    def execution_model_neural_network(self) -> None:
        Console.highlight("7. NEURAL NETWORK TRAINING")
        self.epochs = int(input("Number of epochs: "))
        self.model_nn.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            validation_data=(self.x_val, self.x_val),
            batch_size=self.train_predictive_months,
        )
        print("\nTraining finished")
        Console.stop_continue("[Press Enter to view results]")

    def results_model_neural_network(self) -> None:
        results = self.model_nn.predict(self.x_val)
        plt.scatter(range(len(self.y_val)), self.y_val, c='g')
        plt.scatter(range(len(results)), results, c='r')
        plt.title('Training Results')
        plt.show()
    
    def question_continue_prediction(self) -> None:
        """
        Ask whether to continue training or proceed with prediction.
        """
        Console.highlight("TRAIN AGAIN?")
        print("1. Yes, train again.")
        print("2. No, continue with prediction.")
        response = int(input("=> "))
        self.iterate_train() if response == 1 else self.predict_next_wc()
            
    def predict_next_wc(self) -> None:
        """
        Forecast the next water consumption (wc = water consumption).
        """
        Console.highlight("8. FORECAST NEXT MONTHLY WATER CONSUMPTION")
        self.set_df_predict()
        self.build_matrix_predict()
        self.predict_model()
        print(f"Your next water consumption forecast is: {self.predicted_value} m3")
    
    def set_df_predict(self) -> None:
        """
        Prepare the data used for prediction.
        """
        number_last_months = self.train_predictive_months + 1
        last_months = self.df.tail(number_last_months)
        print(f"\nUsing the last {number_last_months} months of water consumption")
        print(last_months)
        print("\n")
        values = (last_months.values).astype('float32')
        values = values.reshape(-1, 1)
        self.scaler = MinMaxScaler(feature_range=FEATURE_RANGE)
        self.wc_normalize = self.scaler.fit_transform(values)  # Normalize the dataset.
    
    def build_matrix_predict(self) -> None:
        """
        Build the matrix used for prediction.
        """
        self.matrix_sl = self.convert_sequence_to_matrix(1)
        self.matrix_sl.drop(
            self.matrix_sl.columns[[self.train_predictive_months]],
            axis=1,
            inplace=True,
        )
    
    def predict_model(self) -> None:
        """
        Run prediction using the previously trained neural network model.
        """
        values = self.matrix_sl.values
        x_test = values[0:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        results = self.model_nn.predict(x_test)
        predicted = [x for x in results]
        self.predicted_value = round(
            self.scaler.inverse_transform(predicted)[0][0],
            PREDICTION_DECIMAL_PLACES,
        )
