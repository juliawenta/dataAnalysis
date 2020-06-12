import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Source of the data: http://archive.ics.uci.edu/ml/datasets/Air+Quality

data_file = open("data/AirQualityUCI.csv","rt")
data = csv.reader(data_file, delimiter=";")
#print(data.columns.str.cat(sep=";"))
#print(data.head())
#print(data.shape)
next(data)

#Checking how many rows in our data file has missing values (missing values are tagged with "-200")
rows_with_missing_val = 0
for row in data:
    for cell in row:
        if cell == "-200":
            rows_with_missing_val += 1
            break;

print("Rows with missing values: ", rows_with_missing_val)
data_file.close()
"""
Rows with missing values:  8530
"""

air_data = pd.read_csv("data/AirQualityUCI.csv", sep=";") 

#Chcecking types of data:
print(air_data.dtypes)

"""
Date              object
Time              object
CO(GT)            object
PT08.S1(CO)      float64
NMHC(GT)         float64
C6H6(GT)          object
PT08.S2(NMHC)    float64
NOx(GT)          float64
PT08.S3(NOx)     float64
NO2(GT)          float64
PT08.S4(NO2)     float64
PT08.S5(O3)      float64
T                 object
RH                object
AH                object
"""
#Turning attributes into float64 (Date and Time will stay as an object)
air_data["CO(GT)"] = air_data["CO(GT)"].str.replace(",",".")
air_data["C6H6(GT)"] = air_data["C6H6(GT)"].str.replace(",",".")
air_data["T"] = air_data["T"].str.replace(",",".")
air_data["RH"] = air_data["RH"].str.replace(",",".")
air_data["AH"] = air_data["AH"].str.replace(",",".")
air_data[["CO(GT)", "C6H6(GT)", "T", "RH", "AH"]] = air_data[["CO(GT)", "C6H6(GT)", "T", "RH", "AH"]].astype(float)

print(air_data.dtypes)
"""
Date              object
Time              object
CO(GT)           float64
PT08.S1(CO)      float64
NMHC(GT)         float64
C6H6(GT)         float64
PT08.S2(NMHC)    float64
NOx(GT)          float64
PT08.S3(NOx)     float64
NO2(GT)          float64
PT08.S4(NO2)     float64
PT08.S5(O3)      float64
T                float64
RH               float64
AH               float64
dtype: object
"""

#Chcecking missing values in particular data attributes
air_data.isnull().sum()
#print(air_data.isnull().sum())
"""
Date              114
Time              114
CO(GT)            114
PT08.S1(CO)       114
NMHC(GT)          114
C6H6(GT)          114
PT08.S2(NMHC)     114
NOx(GT)           114
PT08.S3(NOx)      114
NO2(GT)           114
PT08.S4(NO2)      114
PT08.S5(O3)       114
T                 114
RH                114
AH                114
Unnamed: 15      9471
Unnamed: 16      9471
dtype: int64
"""

#There are two empty columns "Unnamed", deleting them:
del air_data["Unnamed: 15"]
del air_data["Unnamed: 16"]
#print(air_data.isnull().sum())

"""
Date             114
Time             114
CO(GT)           114
PT08.S1(CO)      114
NMHC(GT)         114
C6H6(GT)         114
PT08.S2(NMHC)    114
NOx(GT)          114
PT08.S3(NOx)     114
NO2(GT)          114
PT08.S4(NO2)     114
PT08.S5(O3)      114
T                114
RH               114
AH               114
dtype: int64
"""

#There is 114 empty cells each column, dropping them:
air_data = air_data.dropna(how="all")
print(air_data.isnull().sum())	

"""
Date             0
Time             0
CO(GT)           0
PT08.S1(CO)      0
NMHC(GT)         0
C6H6(GT)         0
PT08.S2(NMHC)    0
NOx(GT)          0
PT08.S3(NOx)     0
NO2(GT)          0
PT08.S4(NO2)     0
PT08.S5(O3)      0
T                0
RH               0
AH               0
dtype: int64
"""

#Chcecking ststistics of variables - getting min and max value, mean and median of each column
air_data.agg(["min", "max", "mean", "median"])
#print(air_data.agg(["min", "max", "mean", "median"]))

"""
        PT08.S1(CO)     NMHC(GT)  PT08.S2(NMHC)      NOx(GT)  PT08.S3(NOx)     NO2(GT)  PT08.S4(NO2)  PT08.S5(O3)
min     -200.000000  -200.000000    -200.000000  -200.000000   -200.000000 -200.000000   -200.000000  -200.000000
max     2040.000000  1189.000000    2214.000000  1479.000000   2683.000000  340.000000   2775.000000  2523.000000
mean    1048.990061  -159.090093     894.595276   168.616971    794.990168   58.148873   1391.479641   975.072032
median  1053.000000  -200.000000     895.000000   141.000000    794.000000   96.000000   1446.000000   942.000000
"""

air_data.shape
print(air_data.shape)

#Replacing -200 values by missing values:

air_data = air_data.replace("-200.0",np.NaN)
air_data = air_data.replace(-200,np.NaN)
air_data = air_data.replace("-200",np.NaN)

print(air_data.isnull().sum())	

"""
Date                0
Time                0
CO(GT)           1683
PT08.S1(CO)       366
NMHC(GT)         8443
C6H6(GT)          366
PT08.S2(NMHC)     366
NOx(GT)          1639
PT08.S3(NOx)      366
NO2(GT)          1642
PT08.S4(NO2)      366
PT08.S5(O3)       366
T                 366
RH                366
AH                366
dtype: int64
"""

#NMHC (GT) variable is the most incomplete, also NO2(GT), NOx(GT), NMHC(GT), CO(GT) variiables contain a lot of empty values
#Chcecking the correlations:
correlation = air_data.corr()
print(correlation)

"""
                 CO(GT)  PT08.S1(CO)  NMHC(GT)  C6H6(GT)  PT08.S2(NMHC)   NOx(GT)  PT08.S3(NOx)   NO2(GT)  PT08.S4(NO2)  PT08.S5(O3)         T        RH        AH
CO(GT)         1.000000     0.879288  0.889734  0.931078       0.915514  0.795028     -0.703446  0.683343      0.630703     0.854182  0.022109  0.048890  0.048556
PT08.S1(CO)    0.879288     1.000000  0.790670  0.883795       0.892964  0.713654     -0.771938  0.641529      0.682881     0.899324  0.048627  0.114606  0.135324
NMHC(GT)       0.889734     0.790670  1.000000  0.902559       0.877696  0.812685     -0.771135  0.731193      0.853267     0.766723  0.391587 -0.191454  0.269738
C6H6(GT)       0.931078     0.883795  0.902559  1.000000       0.981950  0.718839     -0.735744  0.614474      0.765731     0.865689  0.198956 -0.061681  0.167972
PT08.S2(NMHC)  0.915514     0.892964  0.877696  0.981950       1.000000  0.704435     -0.796703  0.646245      0.777254     0.880578  0.241373 -0.090380  0.186933
NOx(GT)        0.795028     0.713654  0.812685  0.718839       0.704435  1.000000     -0.655707  0.763111      0.233731     0.787046 -0.269683  0.221032 -0.149323
PT08.S3(NOx)  -0.703446    -0.771938 -0.771135 -0.735744      -0.796703 -0.655707      1.000000 -0.652083     -0.538468    -0.796569 -0.145112 -0.056740 -0.232017
NO2(GT)        0.683343     0.641529  0.731193  0.614474       0.646245  0.763111     -0.652083  1.000000      0.157360     0.708128 -0.186533 -0.091759 -0.335022
PT08.S4(NO2)   0.630703     0.682881  0.853267  0.765731       0.777254  0.233731     -0.538468  0.157360      1.000000     0.591144  0.561270 -0.032188  0.629641
PT08.S5(O3)    0.854182     0.899324  0.766723  0.865689       0.880578  0.787046     -0.796569  0.708128      0.591144     1.000000 -0.027172  0.124956  0.070751
T              0.022109     0.048627  0.391587  0.198956       0.241373 -0.269683     -0.145112 -0.186533      0.561270    -0.027172  1.000000 -0.578621  0.656397
RH             0.048890     0.114606 -0.191454 -0.061681      -0.090380  0.221032     -0.056740 -0.091759     -0.032188     0.124956 -0.578621  1.000000  0.167971
AH             0.048556     0.135324  0.269738  0.167972       0.186933 -0.149323     -0.232017 -0.335022      0.629641     0.070751  0.656397  0.167971  1.000000
"""
		
plt.figure(figsize=(20,10))
sns.heatmap(correlation,annot=True, linewidth=0.5)
#plt.show()
plt.savefig("myfig.png") #Save plot as a image - will be included in report

#Filling empty values based on their correlations:

#1. CO(GT)
correlation["CO(GT)"].to_frame().sort_values("CO(GT)")
print(correlation["CO(GT)"].to_frame().sort_values("CO(GT)"))

"""
PT08.S3(NOx)  -0.703446
T              0.022109
AH             0.048556
RH             0.048890
PT08.S4(NO2)   0.630703
NO2(GT)        0.683343
NOx(GT)        0.795028
PT08.S5(O3)    0.854182
PT08.S1(CO)    0.879288
NMHC(GT)       0.889734
PT08.S2(NMHC)  0.915514
C6H6(GT)       0.931078 - largest correlation
CO(GT)         1.000000
"""

air_data["CO(GT)"] = air_data.groupby("C6H6(GT)")["CO(GT)"].apply(lambda x: x.ffill().bfill())
air_data["CO(GT)"] = air_data.groupby("PT08.S1(CO)")["CO(GT)"].apply(lambda x: x.ffill().bfill())
air_data["CO(GT)"].fillna(method="ffill", inplace=True)   

print("Missing values: ", air_data["CO(GT)"].isnull().sum())
#0 missing values for CO(GT), repeating same steps for rest of attributes:

#2. C6H6(GT)
correlation["C6H6(GT)"].to_frame().sort_values("C6H6(GT)")
print(correlation["C6H6(GT)"].to_frame().sort_values("C6H6(GT)"))
"""
PT08.S3(NOx)  -0.735744
RH            -0.061681
AH             0.167972
T              0.198956
NO2(GT)        0.614474
NOx(GT)        0.718839
PT08.S4(NO2)   0.765731
PT08.S5(O3)    0.865689
PT08.S1(CO)    0.883795
NMHC(GT)       0.902559
CO(GT)         0.931078
PT08.S2(NMHC)  0.981950
C6H6(GT)       1.000000
"""
air_data["C6H6(GT)"] = air_data.groupby("CO(GT)")["C6H6(GT)"].apply(lambda x: x.ffill().bfill())
print("Missing values: ", air_data["C6H6(GT)"].isnull().sum())
#0 missing values for C6H6(GT)

#3. NOx(GT)
correlation["NOx(GT)"].to_frame().sort_values("NOx(GT)")
print(correlation["NOx(GT)"].to_frame().sort_values("NOx(GT)"))
"""
PT08.S3(NOx)  -0.655707
T             -0.269683
AH            -0.149323
RH             0.221032
PT08.S4(NO2)   0.233731
PT08.S2(NMHC)  0.704435
PT08.S1(CO)    0.713654
C6H6(GT)       0.718839
NO2(GT)        0.763111
PT08.S5(O3)    0.787046
CO(GT)         0.795028
NMHC(GT)       0.812685
NOx(GT)        1.000000
"""
air_data["NOx(GT)"] = air_data.groupby("CO(GT)")["NOx(GT)"].apply(lambda x: x.ffill().bfill())
print("Missing values: ", air_data["NOx(GT)"].isnull().sum())
#0 missing values for NOx(GT)

#4. C6H6(GT)
correlation["C6H6(GT)"].to_frame().sort_values("C6H6(GT)")
print(correlation["C6H6(GT)"].to_frame().sort_values("C6H6(GT)"))
"""
PT08.S3(NOx)  -0.735744
RH            -0.061681
AH             0.167972
T              0.198956
NO2(GT)        0.614474
NOx(GT)        0.718839
PT08.S4(NO2)   0.765731
PT08.S5(O3)    0.865689
PT08.S1(CO)    0.883795
NMHC(GT)       0.902559
CO(GT)         0.931078
PT08.S2(NMHC)  0.981950
C6H6(GT)       1.000000
"""
air_data["NO2(GT)"] = air_data.groupby("PT08.S5(O3)")["NO2(GT)"].apply(lambda x: x.ffill().bfill())
air_data["NO2(GT)"] = air_data.groupby("CO(GT)")["NO2(GT)"].apply(lambda x: x.ffill().bfill())
print("Missing values: ", air_data["C6H6(GT)"].isnull().sum())
#0 missing values for C6H6(GT)

air_data.fillna(method="ffill", inplace=True)
air_data.shape
print(air_data.isnull().sum())
#No missing values at all

#Saving cleared to csv
df = pd.DataFrame(air_data)
df.to_csv("data/AirQuality_Cleared.csv", index = False, sep = ";")


