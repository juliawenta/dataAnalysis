import pandas as pd
import matplotlib.pyplot as plt

#open file
airData = pd.read_csv('data/AirQuality_Cleared.csv', sep=";")

print("describe: ")
airData.describe()
#print(airData.describe())
"""describe: 
            CO(GT)  PT08.S1(CO)  ...           RH           AH
count  9357.000000  9357.000000  ...  9357.000000  9357.000000
mean      2.091931  1102.730362  ...    48.817431     1.017382
std       1.438469   219.588101  ...    17.354326     0.404829
min       0.100000   647.000000  ...     9.200000     0.184700
25%       1.000000   938.000000  ...    35.400000     0.726200
50%       1.700000  1062.000000  ...    48.900000     0.987500
75%       2.800000  1237.000000  ...    61.900000     1.306700
max      11.900000  2040.000000  ...    88.700000     2.231000
"""
print("info: ")
airData.info()
#print(airData.info())
"""info: 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9357 entries, 0 to 9356
Data columns (total 15 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Date           9357 non-null   object 
 1   Time           9357 non-null   object 
 2   CO(GT)         9357 non-null   float64
 3   PT08.S1(CO)    9357 non-null   float64
 4   NMHC(GT)       9357 non-null   float64
 5   C6H6(GT)       9357 non-null   float64
 6   PT08.S2(NMHC)  9357 non-null   float64
 7   NOx(GT)        9357 non-null   float64
 8   PT08.S3(NOx)   9357 non-null   float64
 9   NO2(GT)        9357 non-null   float64
 10  PT08.S4(NO2)   9357 non-null   float64
 11  PT08.S5(O3)    9357 non-null   float64
 12  T              9357 non-null   float64
 13  RH             9357 non-null   float64
 14  AH             9357 non-null   float64
dtypes: float64(13), object(2)
memory usage: 1.1+ MB
"""
#plots
print("===========")
col_=airData.columns.tolist()[2:]
#for i in airData.columns.tolist()[2:]:
    #airData.plot(x='Date', y=i)
    #plt.savefig("mainInfo"+str(i)+".png")
   # plt.show()

print("===========")
#airData correlation:
airDataCorrel = airData.corr()
#print(airDataCorrel)
"""Correlation:
                 CO(GT)  PT08.S1(CO)  NMHC(GT)  ...         T        RH        AH
CO(GT)         1.000000     0.886114  0.227667  ...  0.025639  0.020122  0.025227
PT08.S1(CO)    0.886114     1.000000  0.239108  ...  0.037046  0.120042  0.121724
NMHC(GT)       0.227667     0.239108  1.000000  ...  0.082679 -0.057676  0.060621
C6H6(GT)       0.932584     0.886325  0.236859  ...  0.189645 -0.054949  0.155825
PT08.S2(NMHC)  0.918386     0.896015  0.237931  ...  0.231083 -0.082087  0.174921
NOx(GT)        0.773677     0.705739  0.116607  ... -0.253366  0.174825 -0.158192
PT08.S3(NOx)  -0.715683    -0.777913 -0.259475  ... -0.132851 -0.060581 -0.216738
NO2(GT)        0.682774     0.655691  0.149796  ... -0.183268 -0.086009 -0.310988
PT08.S4(NO2)   0.631854     0.676413  0.179990  ...  0.558374 -0.015158  0.630272
PT08.S5(O3)    0.858762     0.901460  0.201575  ... -0.044829  0.137821  0.055483
T              0.025639     0.037046  0.082679  ...  1.000000 -0.570775  0.654768
RH             0.020122     0.120042 -0.057676  ... -0.570775  1.000000  0.180512
AH             0.025227     0.121724  0.060621  ...  0.654768  0.180512  1.000000"""


#Date and Time are objects as we check in clearData.py
airData['dateAndTime'] = airData['Date']+' '+airData['Time']
airData['dateAndTime'].head()
#print(airData['dateAndTime'].head())
# now we have date and time together
airData['dateAndTime'] = pd.to_datetime(airData.dateAndTime, format='%d/%m/%Y %H.%M.%S')
airData.dtypes

#preparing data
airData['month'] = airData['dateAndTime'].dt.month
airData['weekday'] = airData['dateAndTime'].dt.weekday
airData['weekdayName'] = airData['dateAndTime'].dt.day_name()
airData['hours'] = airData['dateAndTime'].dt.hour


airDataCorrelCO = airDataCorrel['CO(GT)'].to_frame().sort_values('CO(GT)')
#print(airDataCorrelCO)
#plot for hourlyAverageCO
plt.figure(figsize=(10,8))
airDataCorrelCO.plot(kind='barh', color='cyan')
plt.title('Correlation with the resulting variable: CO ', fontsize=18)
plt.xlabel('Correlation level')
#plt.savefig("Correlation_CO(GT).png")
plt.show()