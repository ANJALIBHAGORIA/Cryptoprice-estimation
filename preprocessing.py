import os
import pickle
import pandas as pd
from functions import *

print('Preprocessing Data .....')

RATIO_TO_PREDICT = 'ETH-USD'
FUTURE_PERIOD_PREDICT = 3

main_df = pd.DataFrame()

ratios = ["BCH-USD", "BTC-USD", "ETH-USD", "LTC-USD"]
for ratio in ratios:
    df = pd.read_csv(f"crypto_data/{ratio}.csv",
                     names=['time', 'low', 'high', 'open', 'close', 'volume'])
    df = df[['time', 'close', 'volume']]
    df.set_index('time', inplace=True)
    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df['future']))

times = sorted(main_df.index.values)
last_5pct = times[-int(0.15 * len(times))]

validation_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

validation_x, validation_y = preprocess_df(validation_df)
train_x, train_y = preprocess_df(main_df)

with open('validation_x.pickle', 'wb') as f:
    pickle.dump(validation_x, f)
    print('Saved validation_x')

with open('validation_y.pickle', 'wb') as f:
    pickle.dump(validation_y, f)
    print('Saved validation_y')

with open('train_x.pickle', 'wb') as f:
    pickle.dump(train_x, f)
    print('Saved train_x')

with open('train_y.pickle', 'wb') as f:
    pickle.dump(train_y, f)
    print('Saved train_y')

print('Done!')
