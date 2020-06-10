import csv
import pandas as pd

airData = pd.read_csv("data/AirQualityUCI.csv")
print(airData.columns.str.cat(sep=", "))
print(airData.head())
print(airData.shape)