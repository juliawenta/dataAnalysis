import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snc
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error


#open file
airData = pd.read_csv('data/AirQuality_Cleared.csv', sep=";")
print(airData.head())

snc.pairplot(airData[["Date","Time","CO(GT)","PT08.S1(CO)","NMHC(GT)","C6H6(GT)","PT08.S2(NMHC)","NOx(GT)","PT08.S3(NOx)","NO2(GT)","PT08.S4(NO2)","PT08.S5(O3)"]],diag_kind = "auto")
plt.savefig("charts/pairplot.png")
plt.show()

C6H6 = airData['C6H6(GT)'] #target

#pearson
print("===pearson===")
airDataQuality=airData.corr('pearson')
print(airDataQuality)
"""
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
AH             0.025227     0.121724  0.060621  ...  0.654768  0.180512  1.000000
"""

abs(airDataQuality['C6H6(GT)']).sort_values(ascending=False)
print(abs(airDataQuality['C6H6(GT)']).sort_values(ascending=False))
"""
C6H6(GT)         1.000000
PT08.S2(NMHC)    0.982485
CO(GT)           0.932584
PT08.S1(CO)      0.886325
PT08.S5(O3)      0.861688
PT08.S4(NO2)     0.756328
PT08.S3(NOx)     0.737702
NOx(GT)          0.703081
NO2(GT)          0.624668
NMHC(GT)         0.236859
T                0.189645
AH               0.155825
RH               0.054949
Name: C6H6(GT), dtype: float64

"""

airData['C6H6(GT)'].value_counts().plot(kind='bar', color='lightblue', figsize=(50,10))
plt.savefig("charts/C6H6(GT).png")
plt.show()

airData2 = airData
#print("===head airData2===")
airData2=airData2.drop('Date',axis=1)
airData2=airData2.drop('Time',axis=1)
airData2=airData2.drop('T',axis=1)
airData2=airData2.drop('RH',axis=1)
airData2=airData2.drop('AH',axis=1)
airData2=airData2.drop('NMHC(GT)',axis=1)
airData2=airData2.drop('C6H6(GT)',axis=1)
airData2.head()
print(airData2.head())
"""      CO(GT)  PT08.S1(CO)  PT08.S2(NMHC)  ...  NO2(GT)  PT08.S4(NO2)  PT08.S5(O3)
0     2.6       1360.0         1046.0  ...    113.0        1692.0       1268.0
1     2.0       1292.0          955.0  ...     92.0        1559.0        972.0
2     2.2       1402.0          939.0  ...    114.0        1555.0       1074.0
3     2.2       1376.0          948.0  ...    122.0        1584.0       1203.0
4     1.6       1272.0          836.0  ...    116.0        1490.0       1110.0
"""
#print("===c6h6 values===")
airData2Values = airData2.values
C6H6 = C6H6.values

X_train, X_test, y_train, y_test = train_test_split(airData2Values, C6H6, test_size=0.3, random_state=0)
linearRegression2 = LinearRegression(normalize=True)
linearRegression2.fit(X_train, y_train)

print("Predicted values:", linearRegression2.predict(X_test))
yNew = linearRegression2.predict(X_test)
yNew.shape

"""Predicted values: [ 3.8  7.2  2.1 ...  6.   8.7 24.8]"""

print("R^2 score: ", linearRegression2.score(X_test, y_test))
plt.scatter(y_test,yNew,color='black')
plt.show()

dataFrameWithTestVariables = pd.DataFrame({' Actual C6H6(GT)': y_test, 'Predicted C6H6(GT)': yNew})
dataFrameWithTestVariables.head(10)
print(dataFrameWithTestVariables.head(10))
dataFrameWithTestVariables.head(50).plot()

plt.savefig("charts/dataFrameWithTestVariables_C6H6(GT).png")
plt.show()

print('Mean Squared Error:     ', metrics.mean_squared_error(y_test, yNew))
print('Mean Absolute Error:    ', metrics.mean_absolute_error(y_test, yNew))
print('Median Absolute Error:  ', metrics.median_absolute_error(y_test, yNew))

