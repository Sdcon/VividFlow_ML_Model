import keras
from keras import Sequential
from keras.metrics import RootMeanSquaredError as RMSE, MeanAbsoluteError as MAE
from keras.layers import LSTM, Dropout, Dense

'''
    Params Explanation
    train: train data
    units: units of LSTM
    dropout: dropout rate
'''

def create_model(train, units, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=False,
                   input_shape=(train.shape[1], train.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(1))

    return model

def train_model_ts(model,
                   x_train, y_train, x_val, y_val,
                   epochs=300,
                   patience=12,
                   batch_size=32):
    
    model.compile(optimizer='SGD',
                  loss='mean_squared_error',
                  metrics=[RMSE(), MAE()])
    
    es = keras.callbacks.EarlyStopping(
                    monitor="val_loss", 
                    min_delta=0, 
                    patience=patience)
    
    history = model.fit(x_train, y_train,
                        shuffle=False, epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        # callbacks=[es], 
                        verbose=1)
    return history
    