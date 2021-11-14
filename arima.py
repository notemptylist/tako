#!/usr/bin/python

import sys
from math import sqrt
import warnings
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from microprediction import MicroReader
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

mr = MicroReader()

def df_from_lagged(name='die.json'):
    """ Turn lagged times and values into a pandas DataFrame
    """
    lagged = mr.get_lagged_values_and_times(name)
    lagged_times,lagged_values = reversed(lagged[1]), reversed(lagged[0])

    df = pd.DataFrame({'Date' : pd.Series(lagged_times), 'y' : pd.Series(lagged_values)})
    df['Date'] = pd.to_datetime(df['Date'], unit='s')
    df = df.set_index(['Date'])
    return df

def plot_df(df, x, y, title="", xlabel='Date', ylabel='value', dpi=100):
    plt.figure(figsize=(16,4), dpi=dpi)
    plt.plot(x, y, "r-")
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

def select_stream():
    prizes = mr.get_prizes()
    import random
    sponsor = random.choice([item['sponsor'] for item in prizes])
    animal = mr.animal_from_code(sponsor)
    animal = 'Emblossom Moth'
    animal = 'Offcast Goose'
    stream = random.choice([k for k,v in mr.get_sponsors().items() if v == animal])
    return stream

def is_stationary(series):
    """
    Test if a series is stationary.

    null hypothesis: series contains a root unit and is non-stationary.

    1. ADF fuller returns p-value < 0.05, reject the null hypothesis : return True
    2. KPSS returns p-value > 0.05, reject the null hypothesis: return True
    """
    adf = adfuller(series.values, autolag='AIC')
    print(f'ADF stats: {adf[0]}')
    print(f'p-value  : {adf[1]}')
    for k,v in adf[4].items():
        print("Critical Values:")
        print(f"    {k},{v}")

    KPSS = kpss(series.values, regression='c')
    print(f"\nKPSS stats: {KPSS[0]}")
    print(f'p-value  : {KPSS[1]}')
    for k,v in KPSS[3].items():
        print("Critical Values:")
        print(f"    {k},{v}")

    return adf[1] < 0.05 and KPSS[1] > 0.05

def is_autocorrelated(series, stream):
    """
    Calculate the ACF and PACF
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf
    dpi = 100
    nlags = 50

    fig = plt.figure(figsize=(16,6), dpi=dpi)
    ax2 = plt.subplot(223)
    plot_acf(series.values, lags=nlags, ax=ax2)
    ax3 = plt.subplot(224)
    plot_pacf(series.values, lags=nlags, ax=ax3)
    ax1 = plt.subplot(211)
    ax1.plot(series.index, series.values, 'r-')
    ax1.set_title(stream)

    return True

def difference(vals, interval=1):
    diffs = list()
    for i in range(interval, len(vals)):
        value = vals[i] - vals[i - interval]
        diffs.append(value)
    return np.array(diffs)

def inverse_differenced(history, yhat, interval=1):
    return yhat + history[-interval]

def test_arima(data, order):
    train_size = int(len(data) * 0.66)
    train, test = data[0:train_size], data[train_size:]
    history = train[:]

    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
        print(f"history: {len(history)}", flush=True)

    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), (0, 0, 0)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    print(f"Testing {order}", flush=True)
                    rmse = test_arima(list(dataset), order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print(f'ARIMA{order} RMSE={rmse}')
                except Exception as ex:
                    raise
    print(f"Best ARIMA{best_cfg} RMSE={best_score}")
    return best_cfg

def main():

    stream = select_stream()
    print(f"Selected stream {stream}", flush=True)
    df = df_from_lagged(stream)
    #print(df)
    #print(f"Stationary: {is_stationary(df['y'])}")

    #plot_df(df, x=df.index, y=df['y'], title=f"{stream}")
    is_autocorrelated(df, stream)


    freq = '5T'
    df.index = df.index.to_period(freq)
    nlags = len(df)
    sample = 225
    train = df[:-sample]
    test = df[-sample:]

    # test for best p d q configuration for this dataset. 
    ps = list(range(0, 12, 2))
    ds = list(range(0,3))
    qs = list(range(0,3))

    warnings.filterwarnings('ignore')
    print("Evaluating...", flush=True)
    best_order = evaluate_models(df['y'].values, ps, ds, qs)
    model = ARIMA(train, order=best_order, freq=freq)
    model_fit = model.fit()
    print(model_fit.summary())
    start_idx = len(train)
    end_idx = start_idx + sample
    df['forecast'] = model_fit.predict(start=start_idx, end=end_idx, dynamic=True)
    df[['y', 'forecast']].plot(figsize=(12, 6))
    plt.show()

    return True

if __name__ == '__main__':
    main()
