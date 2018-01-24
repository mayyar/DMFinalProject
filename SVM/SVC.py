### SVC (not working) ###
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Dataset/preprocessing-cleaning/autos_cleaned.csv', encoding='latin-1')
df = df[df['price'] > 100]

le = LabelEncoder()
ohe = OneHotEncoder()
'''
X_str = df.select_dtypes(include=[object])
X_str = X_str.apply(le.fit_transform)
X_str = ohe.fit_transform(X_str).toarray()
X_row, X_column = X_str.shape
print(X_column)

#y = df['price'].as_matrix()
#print(type(y))
'''

vTypeArr = le.fit_transform(df['vehicleType'].tolist())
#print(vTypeArr)

gearboxArr = le.fit_transform(df['gearbox'].tolist())
#print(gearboxArr)

modelArr = le.fit_transform(df['model'].tolist())
#print(modelArr)

fTypeArr = le.fit_transform(df['fuelType'].tolist())
#print(fTypeArr)

brandArr = le.fit_transform(df['brand'].tolist())
#print(brandArr)

RepDmgArr = le.fit_transform(df['notRepairedDamage'].tolist())
#print(RepDmgArr)


X = []
X_tmp = []
y = []

arrIdx = 0
for index, row in df.iterrows():
    X.append([vTypeArr[arrIdx], row['yearOfRegistration'], gearboxArr[arrIdx], \
              row['powerPS'], modelArr[arrIdx], row['kilometer'], row['monthOfRegistration'],\
              fTypeArr[arrIdx], brandArr[arrIdx], RepDmgArr[arrIdx]])
    y.append(int(row['price']/1000)*1000)
    arrIdx += 1
    #print(X)
    #if index > 100:
    #    break
#print(y)

print("Finished creating X & y")
print("Now splitting to train & test data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

print("Start training...")
clf = SVC(kernel="linear", cache_size=7000)
#time_start = time.clock()
print("Start fitting...")
clf.fit(X_train, y_train)
print("Start predicting...")
y_pred = clf.predict(X_test)
#time_elapsed = (time.clock() - time_start)
print("Calculating scores...")
ac_score = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_true, y_pred)

print("y prediction:", y_pred)
print("Accuracy score: ", ac_score)
print("Confusion Matrix:")
print(matrix)
#print("Computation time: ", time_elapsed)
print("\n")
