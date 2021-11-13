import os
import pandas as pd
from datetime import datetime as dt
from microprediction import MicroCrawler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from private_config import HTTP_HOME


def plotname(name):
    return os.path.join(HTTP_HOME,name)

def arima(data):
    model = ARIMA(data['y'], order=(0, 0, 1))
    model_fit = model.fit()
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)

def df_from_lagged(times, values):
    lagged_times, lagged_values = reversed(times), reversed(values)
    df = pd.DataFrame({'date' : pd.Series(lagged_times), 'y' : pd.Series(lagged_values)})
    df['date'] = pd.to_datetime(df['date'], unit='s')
    df = df.set_index(['date'])
    return df

class SeriesCrawler(MicroCrawler):

    def include_sponsor(self, sponsor=None, **ignore):
        """ Override this as you see fit to select streams for your crawler """
        spons = ['Emblossom Moth', 'Offcast Goose']
        #'e3b1055033076108b4279c473cde3a67', '0ffca579005ef5d8757270f007c4db76']
        return sponsor in spons

    def exclude_stream(self, name=None, **ignore):
        return '~' in name

    def include_stream(self, name=None, **ignore):
        return 'electricity' in name

    def sample(self, lagged_values, lagged_times=None, name=None, delay=None, **ignored):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        sample_size = self.num_predictions
        nlags = len(lagged_times)
        df = df_from_lagged(lagged_times, lagged_values)
        print(df)
        df.plot(y='y', 
            title=f"{delay} {name}",
            label='Lagged Values')
        plt.savefig(plotname('lagged.png'))
        plt.close()
        print(f"--- You have {delay} seconds to return something.")
        before = dt.now()

        # assuming 5min frequency
        freq = "5T"
        df.index = df.index.to_period(freq)
        model = ARIMA(df, order=(3, 2, 1), freq=freq)
        model_fit = model.fit()
        print(df)
        print(model_fit.summary())
        yhats = model_fit.predict(start=nlags-sample_size, end=nlags, dynamic=False)
        yhats.plot(title='Predicted')
        plt.savefig(plotname('predicted.png'))
        plt.close()

        lapsed = dt.now() - before
        print(f"... {lapsed} seconds later.")
        print(f"+++ Returning {sample_size} predictions")
        yhats = yhats.values[-sample_size:]
        print(yhats)
        return yhats