from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import BatchNormalization
import math


# def build_model(BATCH_SIZE, FEATURES_SIZE, OUTPUT_SIZE):
#     model = Sequential()
#     model.add(LSTM(units=128,  activation='relu',
#                    return_sequences=True, input_shape=(BATCH_SIZE, FEATURES_SIZE)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=64, activation='relu', return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units=32, activation='relu', return_sequences=True))
#     model.add(Dropout(0.2))
#     #全连接，输出， add output layer
#     model.add(Dense(OUTPUT_SIZE,activation='sigmoid'))
#     model.summary()
#     model.compile(metrics=['mae'], loss=['mae'], optimizer='adam')
#     return model


# def build_model():
#     # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
#     model = Sequential()
#     model.add(LSTM(4, return_sequences=True,
#               input_shape=(1, 4), activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50, return_sequences=True, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(LSTM(100, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(1, activation='linear'))
#     model.summary()
#     model.compile(metrics=['mae'],loss='mse', optimizer='rmsprop')
#     return model
def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(BatchNormalization(input_shape=(4,),))
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    # model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(metrics=['mae'], loss='mse', optimizer='rmsprop')
    return model

def train_model():


    train = pd.read_csv('./data/clear_train_4.csv')
    test = pd.read_csv('./data/clear_test_4.csv')
    train_t = train[train['pushPrice'] < 150]
    del train_t['pushPrice']
    train_y = train_t['transcycle']

    scaler = MinMaxScaler(feature_range=(0, 1))
    # train_X = scaler.fit_transform(train_t).reshape(-1, 1, 4)
    train_X = scaler.fit_transform(train_t)
    train_Y = np.log1p(train_y)
    test_X = scaler.fit_transform(test)
    model = build_model()
    
    save_path = './weight/weight_lstm.tf'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        save_path, save_format='tf', monitor='mae',verbose=0, save_best_only=True, save_weights_only=True)
    train_tag = False
    if train_tag == True:
        history = model.fit(train_X, train_Y, batch_size=64, epochs=40,
                            validation_split=0.1,
                            callbacks=[checkpoint],
                            verbose=1)
        plt.figure(1)
        plt.plot(history.history['mae'], label='train')
        plt.plot(history.history['val_mae'], label='valid')
        mae_csv = pd.DataFrame(columns=['train_mae','val_mae'])
        mae_csv['train_mae'] = list(history.history['mae'])
        mae_csv['val_mae'] = list(history.history['val_mae'])
        mae_csv.to_csv('./submit/LSTM_train_mae.csv',index=0)
        plt.legend()
        plt.show()

    else:
        model.load_weights(save_path)
        y_p = model.predict(test_X)
        pred = np.expm1(y_p)
        file5 = pd.read_csv('./data/file5.csv')

        submit_file = pd.DataFrame(columns=['id'])
        submit_file['id'] = file5['carid']
        # 向上取整
        submit_file['transcycle'] = [math.ceil(i) for i in list(pred)]
        with open('./submit/附件6：门店交易模型结果-LSTM.txt','a+', encoding='utf-8') as f:
            for line in submit_file.values:
                f.write((str(line[0])+'\t'+str(line[1])+'\n'))

        
    print()



if __name__ == '__main__':
    train_model()
