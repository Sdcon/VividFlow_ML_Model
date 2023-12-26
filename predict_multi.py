import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import *
import argparse
import keras

def rearange(data, predict):
    for i in range(len(data.columns)-1):
        data[i] = data[i+1]
    data[len(data.columns)-1] = predict
    return data
    
# argv
parser = argparse.ArgumentParser()
parser.add_argument('data', help='data for prediction')
parser.add_argument('param', help='parameter file')
parser.add_argument('window_size', help='window_size')
parser.add_argument('next', help='how much does model have to predict')
args = parser.parse_args()

# read data
path = args.data
window_step = int(args.window_size)
data = pd.read_csv(path)
print(f'read data...shape={data.shape}')

# make datetime as index
data['datetime'] = pd.to_datetime(data['date'])+pd.to_timedelta(data['hour'], unit='h')
data = data.drop(['date', 'hour'], axis=1)
data.set_index('datetime', inplace=True)
data.columns = ['amount']
data['amount'] = data['amount'].astype(float)

# generate missing index
full_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='H')
data = data.reindex(full_index)
data['amount'] = data['amount'].bfill()

# normalize data
index = data.index
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)
data = pd.DataFrame(data)
data.set_index(index, inplace=True)

# slice data to be window step
data = one_step_forecast(data, window_step)
x = [f'x_{i}' for i in range (2, window_step+1)]
x.append('y')
data = pd.DataFrame(data[x][-1:])
data.columns = range(len(data.columns))
for i in range(len(data.columns)-1):
    data[i]=data[i+1]
# data[window_step]=0
data.index += pd.Timedelta(hours=window_step+1)

model = keras.models.load_model(f'./param/{args.param}')
for step in range(int(args.next)):
    predict = model.predict(data)
    data = rearange(data, predict)
predict = np.array(data).reshape(-1, 1)
predict = scaler.inverse_transform(predict)
print(predict)