# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clase para entrenar modelo de redes neuronales y predecir el consumo
de agua mensual de una vivienda.
"""
__author__ = "Gonzalo Chacaltana Buleje <gchacaltanab@outlook.com>"
from Forecasting.Models.DataModel.WaterConsumptionDataModel import WaterConsumptionDataModel
from Forecasting.Helpers.Functions import Console
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.layers import Dense, Activation, Flatten
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
import numpy as np  # pip install numpy==1.19.3
import pandas as pd
import matplotlib.pylab as plt
import sys


class WCTrainPredict(object):

    def __init__(self):
        self.df = None
        self.df_normalize = None
        self.building = None
        self.apartment = None
        self.set_config_neural_network()
        self.set_config_plt()

    def set_config_plt(self):
        plt.rcParams['figure.figsize'] = (16, 9)
        plt.style.use('fast')

    def set_config_neural_network(self):
        self.epochs = 60 # (default)
        self.train_predictive_months = 3 # (default)
        # matrix_sl: Matriz de aprendizaje supervisado para TS (Time Series)
        self.matrix_sl = None
        self.train_percentage = 80  # 80% (default)
        self.test_percentage = 20  # 20% (default)
        self.data_train = []
        self.data_test = []
        self.x_train, self.y_train, self.x_val, self.y_val = [], [], [], []
        self.model_nn = None
        self.activation = "tanh" # (default)
        self.optimizer = "Adam" # (default)
        self.loss = "mean_absolute_error" # (default)

    def run(self):
        self.input_data()
        self.build_wc_df()
        self.get_time_series_wc()
        self.normalize_wc()
        self.iterate_train()
    
    def iterate_train(self):
        self.convert_time_series_to_supervised_learning_matrix()
        self.data_partition()
        self.create_model_neural_network()
        self.execution_model_neural_network()
        self.results_model_neural_network()
        self.question_continue_prediction()

    def input_data(self):
        """
        Ingresar información para obtener dataset
        """
        Console.highlight("1. SELECCIONAR CONJUNTO DE DATOS")
        self.building = str(input("Ingrese nombre de edificio: ")).upper()
        self.apartment = str(input("Ingrese nombre de vivienda: ")).upper()

    def build_wc_df(self):
        """
        Construir dataframe de consumo de agua
        """
        Console.highlight("2. CONSUMOS MENSUALES DE AGUA DEL DEPARTAMENTO {}".format(self.apartment))
        self.get_dataset_water_consumption()
        self.set_dataframe_wc()
        self.show_dataframe_wc()
        Console.stop_continue("[Ver consumos del {}]".format(self.apartment))
        print(self.df)
        Console.stop_continue("[Enter para continuar]")

    def get_dataset_water_consumption(self):
        wcdm = WaterConsumptionDataModel()
        self.wm_consumptions = wcdm.get_wm_month_consumption_by_property(self.building, self.apartment)
        if len(self.wm_consumptions) == 0:
            raise Exception("El departamento {} del edificio {} no tiene registros de consumos mensuales.".format(
                self.apartment, self.building))
    
    def set_dataframe_wc(self):
        """
        Crear dataframe para wc (Water Consumption)
        """
        self.df = pd.DataFrame(self.wm_consumptions, columns = ['Anio','Mes','fecha_facturacion', 'm3','Facturado'])
        self.df['fecha_facturacion'] = pd.to_datetime(self.df['fecha_facturacion'])
        self.df = self.df.set_index('fecha_facturacion')
    
    def show_dataframe_wc(self):
        """
        Devolver información del dataset
        """
        print("Observaciones: {} consumos mensuales de agua".format(len(self.df.index)))
        print("Fecha Minima: {}".format(
            (self.df.index.min()).strftime("%d/%m/%Y")))
        print("Fecha Maxima: {}".format(
            (self.df.index.max()).strftime("%d/%m/%Y")))

    def get_time_series_wc(self):
        """
        Obtener y mostrar serie de tiempo
        """
        self.df = self.df.drop(['Anio','Mes','Facturado'], axis=1)
        print("SERIE DE TIEMPO")
        print(self.df)
        Console.stop_continue("[Enter para continuar]")

    def normalize_wc(self):
        """
        Normalizar valores del consumo de m3 a valores de -1 a 1.
        """
        Console.highlight("3. NORMALIZACION DE LOS DATOS")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        self.df_normalize = scaler.fit_transform(self.df.values.reshape(-1, 1))
        print("Datos de la serie de tiempo normalizados [-1,1].")
        print(self.df_normalize)
        Console.stop_continue("[Enter para continuar]")

    def convert_time_series_to_supervised_learning_matrix(self):
        """
        Convertimos serie de tiempo en una matrix de aprendizaje supervisado, con tantas variables predictoras
        definidas por el atributo train_predictive_month
        """
        Console.highlight("4. CONVERTIR SERIE DE TIEMPO EN MATRIZ DE APRENDIZAJE SUPERVISADO")
        self.train_predictive_months = int(input("Ingresar la cantidad de variables predictoras: "))
        self.matrix_sl = self.convert_sequence_to_matrix(1, 1)
        print("Matriz de la serie de tiempo con {} variables predictoras (entrenamiento)".format(self.train_predictive_months))
        print("\n")
        print(self.matrix_sl)
        Console.stop_continue("[Enter para continuar]")

    def convert_sequence_to_matrix(self, n_out=1, dropnan=True):
        n_vars = 1 if type(
            self.df_normalize) is list else self.df_normalize.shape[1]
        df = pd.DataFrame(self.df_normalize)
        cols, names = list(), list()
        # Agregando las variables predictoras
        for i in range(self.train_predictive_months, 0, -1):
            cols.append(df.shift(i))
            names += [('v_pred(t-%d)' % (i))]

        # Agregando variable objetivo y(t)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('y(t)')]
            else:
                names += [('v_y_%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # Elimina registros con valores NaN
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def data_partition(self):
        Console.highlight("5. PARTICION DEL CONJUNTO DE DATOS")
        self.input_data_partition()
        self.set_data_partition()
        self.build_xy_train_test()
        Console.stop_continue("[Enter para continuar]")

    def input_data_partition(self):
        self.train_percentage = int(input("Ingresar porcentaje para datos de entrenamiento: "))
        self.test_percentage = int(input("Ingresar porcentaje para datos de pruebas: "))
        percentage_sum = self.train_percentage + self.test_percentage
        if percentage_sum!=100:
            raise Exception("La suma de los porcentajes de entrenamiento y pruebas es invalido")

    def set_data_partition(self):
        values = self.matrix_sl.values
        total_rows = len(values)
        self.n_train = int(round(total_rows*(self.train_percentage/100), 0))
        self.n_test = total_rows - self.n_train
        self.data_train = values[:self.n_train, :]
        self.data_test = values[self.n_train:, :]

    def build_xy_train_test(self):
        self.x_train, self.y_train = self.data_train[:,:-1], self.data_train[:, -1]
        self.x_val, self.y_val = self.data_test[:, :-1], self.data_test[:, -1]
        self.x_train = self.x_train.reshape(
            (self.x_train.shape[0], 1, self.x_train.shape[1]))
        self.x_val = self.x_val.reshape((self.x_val.shape[0], 1, self.x_val.shape[1]))
        print("\n")
        print("ENTRENAMIENTO ({} %) : {} observaciones (meses)".format(
            (self.train_percentage), self.n_train))
        print("PRUEBAS ({} %) : {} observaciones (meses)".format(
            (self.test_percentage), self.n_test))

    def create_model_neural_network(self):
        """
        Creamos modelo de red neuronal feed forward.
        Arquitectura.
        01 capa oculta con "n" neuronas.
        Función de activación: Tangente Hiperbólica.(Valores -1 a 1)
        Optimizador: Adam
        Métrica de Pérdida: (Loss) Error Absoluto Medio
        Para calcular el acuracy, se utilizará  Error Cuadrático Medio (MSE)
        """
        Console.highlight("6. CREANDO LA RED NEURONAL")
        print("+ 01 capa oculta de n neuronas")
        print("+ Salida: 01 neurona")
        print("+ Funcion de activacion: Tangente Hiperbolica")
        print("+ Optimizador: Adam")
        print("+ Funcion de perdida: Error Absoluto Medio (MAE)")
        print("+ Calculo del Acuracy: Error Cuadratico Medio (MSE)")
        print("\n\n")
        self.neurals_hidden_layer = int(input("Neuronas para la capa de entrada: "))
        print("\n")
        self.model_nn = Sequential()
        self.model_nn.add(Dense(self.neurals_hidden_layer, input_shape = (1,self.train_predictive_months), activation = self.activation))
        self.model_nn.add(Flatten())
        self.model_nn.add(Dense(1, activation = self.activation))
        self.model_nn.compile(loss=self.loss, optimizer=self.optimizer, metrics=["mse"])
        self.model_nn.summary()
        Console.stop_continue("[Enter para continuar]")

    def execution_model_neural_network(self):
        Console.highlight("7. ENTRENAMIENTO DE LA RED NEURONAL")
        self.epochs = int(input("Numero de epochs: "))
        self.model_nn.fit(self.x_train, self.y_train, epochs=self.epochs, validation_data=(
            self.x_val, self.x_val), batch_size=self.train_predictive_months)
        print("\nFin de entrenamiento")
        Console.stop_continue("[Enter para ver resultado]")

    def results_model_neural_network(self):
        results = self.model_nn.predict(self.x_val)
        plt.scatter(range(len(self.y_val)), self.y_val, c='g')
        plt.scatter(range(len(results)), results, c='r')
        plt.title('Resultados de Entrenamiento')
        plt.show()
    
    def question_continue_prediction(self):
        """
        Muestra texto con pregunta a fin de confirmar si se continua
        con el entrenamiento o si se va realizar la predicción
        """
        Console.highlight("VOLVER A ENTRENAR?")
        print("1. Si, volver a entrenar.")
        print("2. No, continuar con la prediccion.")
        response = int(input("=> "))
        self.iterate_train() if response == 1 else self.predict_wc()
            
    def predict_wc(self):
        """
        Realiza la predicción del próximo consumo.
        """
        Console.highlight("8. PREDECIR PROXIMO CONSUMO MENSUAL DE AGUA")
        self.set_df_predict()
        self.matrix_sl = self.convert_sequence_to_matrix(1, 1)
        self.matrix_sl.drop(self.matrix_sl.columns[[self.train_predictive_months]], axis = 1, inplace = True)
        values = self.matrix_sl.values
        x_test = values[0:, :]
        x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
        results = self.model_nn.predict(x_test)
        adimen = [x for x in results]    
        inverted = self.scaler.inverse_transform(adimen)
        print("El pronostico de su proximo consummo de agua es : {} m3".format(round(inverted[0][0],3)))
    
    def set_df_predict(self):
        number_last_months = self.train_predictive_months + 1
        last_months = self.df.tail(number_last_months)
        print("\nTomando los ultimos {} meses de consumo de agua".format(number_last_months))
        print(last_months)
        print("\n")
        values = (last_months.values).astype('float32')
        values = values.reshape(-1,1)
        self.scaler = MinMaxScaler(feature_range = (-1, 1))
        self.df_normalize = self.scaler.fit_transform(values) # Normalizar conjunto de datos.
    
