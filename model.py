import time
import pickle
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dropout, Dense, BatchNormalization, CuDNNLSTM

EPOCHS = 50
BATCH_SIZE = 64
RATIO_TO_PREDICT = 'ETH-USD'
NAME = f"LSTM model predicts {RATIO_TO_PREDICT} prices-{int(time.time())}"

with open('validation_x.pickle', 'rb') as f:
    validation_x = pickle.load(f)

with open('validation_y.pickle', 'rb') as f:
    validation_y = pickle.load(f)

with open('train_x.pickle', 'rb') as f:
    train_x = pickle.load(f)

with open('train_y.pickle', 'rb') as f:
    train_y = pickle.load(f)

model = Sequential()

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(validation_x, validation_y))
