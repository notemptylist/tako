import os
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
from datetime import datetime as dt
from microprediction import MicroCrawler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from private_config import HTTP_HOME


def plotname(name):
    return os.path.join(HTTP_HOME,name)

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
        history.append(test.iloc[t])

    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

def df_from_lagged(times, values):
    lagged_times, lagged_values = reversed(times), reversed(values)
    df = pd.DataFrame({'date' : pd.Series(lagged_times), 'y' : pd.Series(lagged_values)})
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.set_index(['date'])
    return df

def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), (0, 0, 0)
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    rmse = test_arima(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print(f'ARIMA{order} RMSE={rmse}')
                except Exception as ex:
                    raise
    print(f"Best ARIMA{best_cfg} RMSE={best_score}")
    return best_cfg

class SeriesCrawler(MicroCrawler):

    def include_sponsor(self, sponsor=None, **ignore):
        """ Override this as you see fit to select streams for your crawler """
        spons = ['Emblossom Moth', 'Offcast Goose']
        #'e3b1055033076108b4279c473cde3a67', '0ffca579005ef5d8757270f007c4db76']
        return sponsor in spons

    def exclude_stream(self, name=None, **ignore):
        return '~' in name

#    def include_stream(self, name=None, **ignore):
#        return 'electricity' in name


    def sample(self, lagged_values, lagged_times=None, name=None, delay=None, **ignored):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        sample_size = self.num_predictions
        nlags = len(lagged_times)
        df = df_from_lagged(lagged_times, lagged_values)

        # debug info 
        print(df)
        df.plot(y='y', 
            title=f"{delay} {name}",
            label='Lagged Values')
        plt.savefig(plotname('lagged.png'))
        plt.close()

        before = dt.now()
        # assuming 5min frequency
        freq = "5T"
        df.index = df.index.to_period(freq)

        # split training data
        train = df[:-sample_size]
        test = df[-sample_size:]

        # test for best p d q configuration for this dataset. 
        ps = range(0, 12, 2)
        ds = range(0,3)
        qs = range(0,3)

        best_order = evaluate_models(train, ps, ds, qs)
        model = ARIMA(train, order=best_order, freq=freq)
        model_fit = model.fit()
        print(model_fit.summary())
        yhats = model_fit.predict(start=nlags, end=nlags+sample_size, dynamic=False)
        yhats.plot(title='Predicted')
        plt.savefig(plotname('predicted.png'))
        plt.close()

        lapsed = dt.now() - before
        print(f"... {lapsed} seconds later.")
        print(f"+++ Returning {sample_size} predictions")
        yhats = yhats.values[-sample_size:]
        print(yhats)
        return yhats