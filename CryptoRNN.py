import pandas as pd
import numpy as np
import keras
from keras.layers import Dense, Dropout,LSTM, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import TensorBoard,ModelCheckpoint
import tensorflow as tf 
import warnings
from sklearn.preprocessing import scale
from collections import deque
import random
import time
warnings.filterwarnings('ignore')


data = pd.DataFrame()
ratios = ['BCH-USD','BTC-USD','ETH-USD','LTC-USD']

RATIO_TO_PREDICT = 'LTC-USD'
FUTURE_PREIOD_PREDICT = 3
SEQ_LEN = 60

EPOCHS = 3
BATCH_SIZE = 64
NAME = f"{SEQ_LEN}-SEQ-{EPOCHS}-EPOCHS-{FUTURE_PREIOD_PREDICT}-FRP-{int(time.time())}"


def preprocessing_data(data): # percentage change, scaling, Sequencing and shuffling
        
    data = data.drop('future', axis = 1) # dont need anymore

    for col in data.drop('target', axis=1).columns:
        data[col] = data[col].pct_change(1)
        data.dropna(inplace=True)

        data[col] = scale((data[col].values))
    
    data.dropna(inplace=True)

    sequential_data = []
    prev_data = deque(maxlen=SEQ_LEN)
    for x in data.values:
        prev_data.append([n for n in x[:-1]])
        if (len(prev_data) == SEQ_LEN):
            sequential_data.append([np.array(prev_data), x[-1]])
    
    random.shuffle(sequential_data)

    buy = []        # 1
    sell = []       # 0

    for seq, target in sequential_data:
        if target==1:
            buy.append([seq, target])
        else:
            sell.append([seq,target])

    random.shuffle(buy)
    random.shuffle(sell)

    lower = min(len(buy), len(sell))

    buy = buy[:lower]
    sell = sell[:lower]

    sequential_data = buy + sell

    random.shuffle(sequential_data)

    x = []
    y = []

    for seq, target in sequential_data:
        x.append(seq)
        y.append(target)

    return np.array(x) , y



def decision_to_purchase(current, future):
    if future > current:
        return 1        # buy
    else:
        return 0        # sell


for ratio in ratios:
    cols = ['time','low','high','open','close','volume']
    path = f"./data/{ratio}.csv"
    
    df = pd.read_csv(path,names=cols)
    
    df.rename(columns={'close': f'{ratio}_close', 'volume': f'{ratio}_volume'},inplace=True)
    
    df.set_index('time',inplace=True)
    
    df = df[[f'{ratio}_close',f'{ratio}_volume']]
    
    if len(data) == 0:
        data = df
    else:
        data = data.join(df)

data.fillna(method='ffill', inplace=True)
data.dropna(inplace=True)

data['future'] = data[f'{RATIO_TO_PREDICT}_close'].shift(periods = -FUTURE_PREIOD_PREDICT)

data['target'] = list(map(decision_to_purchase, data[f'{RATIO_TO_PREDICT}_close'],data['future']))

times = sorted(data.index.values)
pct_20 = sorted(data.index.values)[-int(0.20*len(times))]

train = data[(data.index.values >= pct_20)]
test = data[(data.index.values) < pct_20]

x_train, y_train = preprocessing_data(train) #pct_chng, scaling, sequencing, shuffling (repeatingly), balancing buy/sell
x_test, y_test =  preprocessing_data(test)

print(x_train.shape)

model = Sequential()

model.add(LSTM(128, activation='relu', return_sequences=True , input_shape = (x_test.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,activation='relu', return_sequences = False))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(2, activation = 'softmax'))

adam = Adam(lr=0.001, decay=1e-6)
model.compile(optimizer=adam , metrics=['accuracy'], loss='sparse_categorical_crossentropy')

tensorboard = TensorBoard(log_dir='./logs/{}'.format(NAME))
filename = "RNN_FINAL-{epoch:01d}-{val_acc:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filename , monitor='val_acc', verbose = 1, save_best_only=True , mode='max'))

# >checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max'))

print(x_train.shape)
print(len(y_train))
history = model.fit(x_train,y_train
                    ,epochs=EPOCHS, 
                    callbacks= [tensorboard, checkpoint], 
                    validation_data=(x_test, y_test) 
                    ,batch_size=BATCH_SIZE
                    )


evaluation = model.evaluate(x_test, y_test , verbose=0)

print("Evaluation result: " , evaluation)

model.save(filepath="./models/{}".format(NAME))

