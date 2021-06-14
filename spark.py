import pandas as pd
from pyspark.sql import SparkSession
# from plots import draw_histogram

spark = SparkSession \
    .builder \
    .appName("minadzb") \
    .getOrCreate()

data_columns_classification = ['Date', 'Location', 'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
                               'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am',
                               'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am',
                               'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday']
data_columns_regression = ['Voltage', 'Global_reactive_power',
                           'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']


def normalize_data(input_data, row=''):
    print(row)
    average_value = sum([float(item) for item in input_data]) / len(input_data)
    standard_deviation = (sum([(float(item) - average_value) ** 2 for item in input_data]) / len(input_data)) ** 0.5
    print('sd: ' + str(standard_deviation) + '; av: ' + str(average_value))
    if standard_deviation == 0.0:
        return input_data

    return list([(float(item) - average_value) / standard_deviation for item in input_data])


def map_and_convert_column(column, dataset=None):
    if dataset is None:
        dataset = {}
    counter = 0
    for index in range(len(column)):
        if column[index] not in dataset:
            dataset[column[index]] = counter
            column[index] = counter
            counter += 1
        else:
            column[index] = dataset[column[index]]
    return column


# label - Global_intensity -  minute-averaged voltage (in volt)
def load_power_as_df(file_path):
    df_normalized = pd.read_csv('data/power-formatted.csv')
    # df = pd.read_csv(file_path, names=['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
    #                                    'label', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'],
    #                  skiprows=1, sep=';')
    # df = df.drop(columns=['Date', 'Time'])
    # df = df.dropna()
    #
    # for index in range(len(df.columns)):
    #     draw_histogram(xs=df[df.columns[index]], title=df.columns[index])
    #
    # df['label'] = normalize_data(df['label'])
    # df['Global_reactive_power'] = normalize_data(df['Global_reactive_power'])
    # df['Voltage'] = normalize_data(df['Voltage'])
    # df['Global_active_power'] = normalize_data(df['Global_active_power'])
    # df['Sub_metering_1'] = normalize_data(df['Sub_metering_1'])
    # df['Sub_metering_2'] = normalize_data(df['Sub_metering_2'])
    # df['Sub_metering_3'] = normalize_data(df['Sub_metering_3'])
    # df.to_csv('data/power-formatted.csv')

    spark_df = spark.createDataFrame(df_normalized)
    spark_df.createOrReplaceTempView('power')

    return spark_df


def load_weather_as_df(file_path):
    df_normalized = pd.read_csv('data/weather-formatted.csv')
    # df = pd.read_csv(file_path)
    # wind_dir_map = {}
    # date_map = {}
    #
    # def convert_date(column):
    #     for index in range(len(column)):
    #         column[index] = column[index][5:]
    #     return column
    #
    # def convert_raining(column):
    #     for index in range(len(column)):
    #         column[index] = 0 if column[index] == 'No' else 1
    #     return column

    # df['Date'] = normalize_data(map_and_convert_column(convert_date(df['Date']), date_map), 'Date')
    # df['MinTemp'] = normalize_data(df['MinTemp'].fillna(method="bfill").fillna(method="ffill"), 'MinTemp')
    # df['MaxTemp'] = normalize_data(df['MaxTemp'].fillna(method="bfill").fillna(method="ffill"), 'MaxTemp')
    # df['Rainfall'] = normalize_data(df['Rainfall'].fillna(method="bfill").fillna(method="ffill"), 'Rainfall')
    # df['Evaporation'] = normalize_data(df['Evaporation'].fillna(method="bfill").fillna(method="ffill"), 'Evaporation')
    # df['Sunshine'] = normalize_data(df['Sunshine'].fillna(method="bfill").fillna(method="ffill"), 'Sunshine')
    # df['WindGustDir'] = map_and_convert_column(df['WindGustDir'].fillna(method="ffill"), wind_dir_map)
    # df['Location'] = map_and_convert_column(df['Location'])
    # df['WindGustSpeed'] = normalize_data(df['WindGustSpeed'].fillna(method="bfill").fillna(method="ffill"), 'WindGustSpeed')
    # df['WindDir9am'] = map_and_convert_column(df['WindDir9am'].fillna(method="bfill").fillna(method="ffill"), wind_dir_map)
    # df['WindDir3pm'] = map_and_convert_column(df['WindDir3pm'].fillna(method="bfill").fillna(method="ffill"), wind_dir_map)
    # df['WindSpeed9am'] = normalize_data(df['WindSpeed9am'].fillna(method="bfill").fillna(method="ffill"), 'WindSpeed9am')
    # df['WindSpeed3pm'] = normalize_data(df['WindSpeed3pm'].fillna(method="bfill").fillna(method="ffill"), 'WindSpeed3pm')
    # df['Humidity9am'] = normalize_data(df['Humidity9am'].fillna(method="bfill").fillna(method="ffill"), 'Humidity9am')
    # df['Humidity3pm'] = normalize_data(df['Humidity3pm'].fillna(method="bfill").fillna(method="ffill"), 'Humidity3pm')
    # df['Pressure9am'] = normalize_data(df['Pressure9am'].fillna(method="bfill").fillna(method="ffill"), 'Pressure9am')
    # df['Pressure3pm'] = normalize_data(df['Pressure3pm'].fillna(method="bfill").fillna(method="ffill"), 'Pressure3pm')
    # df['Cloud9am'] = normalize_data(df['Cloud9am'].fillna(method="bfill").fillna(method="ffill"), 'Cloud9am')
    # df['Cloud3pm'] = normalize_data(df['Cloud3pm'].fillna(method="bfill").fillna(method="ffill"), 'Cloud3pm')
    # df['Temp9am'] = normalize_data(df['Temp9am'].fillna(method="bfill").fillna(method="ffill"), 'Temp9am')
    # df['Temp3pm'] = normalize_data(df['Temp3pm'].fillna(method="bfill").fillna(method="ffill"), 'Temp3pm')
    # df['RainToday'] = convert_raining(df['RainToday'].fillna(method="bfill").fillna(method="ffill"))
    # df['label'] = convert_raining(df['label'].fillna(method="bfill").fillna(method="ffill"))
    # df.to_csv('data/weather-formatted.csv')
    # for index in range(len(df.columns)):
    #     draw_histogram(xs=df[df.columns[index]], title=df.columns[index])
    #
    # for index in range(len(df_normalized.columns)):
    #     draw_histogram(xs=df_normalized[df_normalized.columns[index]], title=df_normalized.columns[index])

    spark_df = spark.createDataFrame(df_normalized)
    spark_df.createOrReplaceTempView('weather')
    spark_df.show()
    return spark_df


def load_data_into_spark_regression():
    return load_power_as_df("data/household_power_consumption.txt")


def load_data_into_spark_classification():
    return load_weather_as_df("data/weatherAUS.csv")
