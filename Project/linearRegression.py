import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dataAnalysis import airData

#functions
def correlationFunction(x, y, timeInHours=0):
    return x.corr(y.shift(timeInHours))

# n - number of shifts
def checkShift(x, y, n):
    correlationX = [correlationFunction(x, y, timeInHours=i) for i in range(n)]
    newArray = pd.DataFrame(list(correlationX)).reset_index()
    newArray.rename(columns={0: 'correlation', 'index': 'Shift_num'}, inplace=True)
    #print("")
    newArray['correlationFix'] = newArray['correlation'].abs()
    shift = newArray.loc[newArray['correlationFix'] == newArray['correlationFix'].max(), 'Shift_num']
    tmpShift = shift.to_frame()
    shift = tmpShift.Shift_num.max()

    return shift

#We are now creating a new DataFrame with a 12-hour shift
def shiftDataFrame12(airData, target=None, timeInHours=0):
    newSchift = {}
    if not timeInHours and not target:
        return airData
    for i in airData.columns:
        if i == target:
            newSchift[i] = airData[target]
        else:
            newSchift[i] = airData[i].shift(periods=timeInHours)
    #return new dataFrame
    return  pd.DataFrame(data=newSchift)


x = airData.RH       # independent variable
y = airData['CO(GT)']    # dependent variable
N = 20           # number of shifts who will be checked

checkShiftRH = checkShift(x,y,N)
print('Optimal shift for RH: ', checkShiftRH)

correlationFunction(x, y, timeInHours=checkShiftRH)
print(correlationFunction(x, y, timeInHours=checkShiftRH))

#We check a variable with very low correlation with the resulting CO (GT) variable
x = airData.AH       # independent variable
checkShiftAH = checkShift(x,y,N)
print('Optimal shift for AH: ',checkShiftAH)

correlationFunction(x, y, timeInHours=checkShiftAH)
print(correlationFunction(x, y, timeInHours=checkShiftAH))

#We check a variable with very low correlation with the resulting CO (GT) variable.
x = airData['T']      # independent variable
checkShiftT = checkShift(x,y,N)
print('Optimal shift for T: ',checkShiftT)

correlationFunction(x, y, timeInHours=checkShiftT)
print(correlationFunction(x, y, timeInHours=checkShiftT))

#preparing data for model
airData2 = airData[['dateAndTime', 'CO(GT)','RH', 'T']]
airData2['weather_time'] = airData2['dateAndTime']

airData2.head(3)
print(airData2.head(3))

airData3 = shiftDataFrame12(airData2, 'weather_time', timeInHours=12)
airData3.rename(columns={'weather_time':'Shift_weather_time'}, inplace=True)
print("  ")
#airData3.head(13)
#print(airData3.head(13))

airData4 = shiftDataFrame12(airData3, 'RH', timeInHours=12)
airData4.rename(columns={'RH':'Shift_RH'}, inplace=True)

airData5 = shiftDataFrame12(airData4, 'T', timeInHours=12)
airData5.rename(columns={'T':'Shift_T'}, inplace=True)
print("  airData5")
airData5 = airData5.dropna(how ='any')
airData5.head()
print(airData5)

airData5.plot(x='Shift_T', y='CO(GT)', style='o')
plt.title('Shift_T vs CO')
plt.xlabel('Shift_T')
plt.ylabel('CO')
#plt.savefig("T_CO(GT).png")
plt.show()

airData5.plot(x='Shift_RH', y='CO(GT)', style='o')
plt.title('Shift_RH vs CO')
plt.xlabel('Shift_RH')
plt.ylabel('CO')
#plt.savefig("RH_CO(GT).png")
plt.show()

#model building
X = airData5[['Shift_RH', 'Shift_T']].values
y = airData5['CO(GT)'].values

# divide - training variables and test variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

trainRegression = LinearRegression()
trainRegression.fit(X_train, y_train)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
print("Predicted values:", trainRegression.predict(X_test))
yNew = trainRegression.predict(X_test)
yNew = np.round(yNew, decimals=2)

#with test variables
dataFrameWithTestVariables = pd.DataFrame({' Actual CO(GT)': y_test, 'Predicted CO(GT)': yNew})
dataFrameWithTestVariables.head(10)
print(dataFrameWithTestVariables.head(10))
"""    Actual CO(GT)  Predicted CO(GT)
0             0.5              1.63
1             1.9              1.91
2             3.4              2.40
3             1.2              1.45
4             2.4              2.40
5             1.3              2.33
6             1.9              1.64
7             4.6              2.55
8             1.3              1.64
9             3.1              2.35

"""

dataFrameWithTestVariables.head(50).plot()
plt.savefig("dataFrameWithTestVariables_CO(GT).png")
plt.show()
print("R^2 score for liner regression: ", trainRegression.score(X_test, y_test))
#info
print('Mean Squared Error:     ', metrics.mean_squared_error(y_test, yNew))
print('Mean Absolute Error:    ', metrics.mean_absolute_error(y_test, yNew))
print('Median Absolute Error:  ', metrics.median_absolute_error(y_test, yNew))


# Plots for month, weekDay and hours level of CO
airData.pivot_table(index='month', values='CO(GT)', aggfunc='mean').plot(kind='bar')
#plt.savefig("month_CO(GT).png") #Save plot as a image - will be included in report
airData.pivot_table(index='weekdayName', values='CO(GT)', aggfunc='mean').plot(kind='bar', color='g')
#plt.savefig("weekDay_CO(GT).png")
airData.pivot_table(index='hours', values='CO(GT)', aggfunc='mean').plot(kind='bar', color='y')
#plt.savefig("hours_CO(GT).png")
plt.show()

# Plots for month, weekDay and hours level of NMHC
airData.pivot_table(index='month', values='NMHC(GT)', aggfunc='mean').plot(kind='bar')
#plt.savefig("month_NMHC(GT).png") #Save plot as a image - will be included in report
airData.pivot_table(index='weekdayName', values='NMHC(GT)', aggfunc='mean').plot(kind='bar', color='g')
#plt.savefig("weekDay_NMHC(GT).png")
airData.pivot_table(index='hours', values='NMHC(GT)', aggfunc='mean').plot(kind='bar', color='y')
#plt.savefig("hours_NMHC(GT).png")
plt.show()

# Plots for month, weekDay and hours level of NO2
airData.pivot_table(index='month', values='NO2(GT)', aggfunc='mean').plot(kind='bar')
#plt.savefig("month_NO2(GT).png") #Save plot as a image - will be included in report
airData.pivot_table(index='weekdayName', values='NO2(GT)', aggfunc='mean').plot(kind='bar', color='g')
#plt.savefig("weekDay_NO2(GT).png")
airData.pivot_table(index='hours', values='NO2(GT)', aggfunc='mean').plot(kind='bar', color='y')
#plt.savefig("hours_NO2(GT).png")
plt.show()

# Plots for month, weekDay and hours level of C6H6
airData.pivot_table(index='month', values='C6H6(GT)', aggfunc='mean').plot(kind='bar')
#plt.savefig("month_C6H6(GT)).png") #Save plot as a image - will be included in report
airData.pivot_table(index='weekdayName', values='C6H6(GT)', aggfunc='mean').plot(kind='bar', color='g')
#plt.savefig("weekDay_C6H6(GT)).png")
airData.pivot_table(index='hours', values='C6H6(GT)', aggfunc='mean').plot(kind='bar', color='y')
#plt.savefig("hours_C6H6(GT)).png")
plt.show()