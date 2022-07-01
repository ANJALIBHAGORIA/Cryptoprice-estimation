import random
import numpy as np
import pandas as pd
from collections import deque
from sklearn import preprocessing

SEQ_LEN = 60


def classify(current, future):
    if future > current:
        return 1
    else:
        return 0


def preprocess_df(df):
    df = df.drop('future', 1)

    for col in df.columns:
        if col != 'target':
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
        df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append(i[:-1])
        if len(prev_days) == 60:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []
    for X, y in sequential_data:
        if y == 1:
            buys.append([X, y])
        elif y == 0:
            sells.append([X, y])

    lowest_len = min(len(buys), len(sells))
    buys = buys[:lowest_len]
    sells = sells[:lowest_len]

    full_df = buys + sells
    random.shuffle(full_df)

    labels = []
    features = []

    for X, y in full_df:
        labels. append(X)
        features.append(y)

    return np.array(labels), features
