import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.metrics import r2_score
model = pickle.load(open('model.pkl', 'rb'))

df = pd.read_csv("house_data.csv")

columns = ['bedrooms', 'floors', 'yr_built', 'price']
df = df[columns]
X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)

arr = np.array(X_test)
arr = arr.astype(np.float64)

pred = model.predict(arr)
y_test = np.array(y_test)
print(y_test.shape, pred.shape)

with open("metrics.txt", "w+") as f:
    f.write(str(pred))



