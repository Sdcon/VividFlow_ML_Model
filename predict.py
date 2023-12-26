import numpy as np
import keras
import argparse
from sklearn.preprocessing import StandardScaler
import pandas as pd
from model import *
from utils import *

# argv
parser = argparse.ArgumentParser()
parser.add_argument('data', help='data for prediction')
parser.add_argument('param', help='parameter file')
parser.add_argument('window_size', help='window_size')
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
data = np.expand_dims(data[x], axis=-1)

# predict
model = keras.models.load_model(f'./param/{args.param}')
predict = model.predict(data)

# inverse value
predict = np.array(predict).reshape(-1, 1)
predict = scaler.inverse_transform(predict)
print(predict[-1])