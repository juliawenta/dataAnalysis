import csv
import numpy as np
import matplotlib.pyplot as plt


plik = open("data/AirQualityUCI.csv",'rt')
airData = csv.reader(plik, delimiter=';')
#print(airData.columns.str.cat(sep=";"))
#print(airData.head())
#print(airData.shape)
next(airData)
arrDate = []
arrTime =[]

y=0
for i in airData:
    for j in i:
        if j == "-200":
            y=y+1
            break;



print(y)

plik.close()
print(" ")

