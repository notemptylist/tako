# prophet based crawler
import os
import sys
import time
import json
import numpy as np
import pandas as pd
from random import sample, choice
from datetime import datetime as dt
import matplotlib.pyplot as plt
from prophet import Prophet
from microprediction import MicroCrawler
from private_config import HTTP_HOME


def lagged_to_frame(times, values):
    frame = { 'ds' : pd.Series(times),
               'y' : pd.Series(values),
            }
    df = pd.DataFrame(frame)
    df['ds'] = pd.to_datetime(df['ds'], unit='s', infer_datetime_format=True)
    df['ds'] = df['ds'].dt.strftime('%Y-%m-%d %H:%M:%S')
    return df

def plotname(name):
    return os.path.join(HTTP_HOME,name)
    """ Simply write out the plot into a {name}.png at the
        HTTP_HOME location
    """

class ProphetCrawler(MicroCrawler):
    """
    """

    def __init__(self, mine=False, **kwargs):
        super().__init__(**kwargs)
        self.mine = mine
        self.mining_time = 0
        self.mr = dict()

    def include_sponsor(self, sponsor=None, **ignore):
        """ Override this as you see fit to select streams for your crawler """
        spons = ['Emblossom Moth', 'Offcast Goose']
        #'e3b1055033076108b4279c473cde3a67', '0ffca579005ef5d8757270f007c4db76']
        return sponsor in spons

    def include_stream(self, name=None, **ignore):
        t = 'electricity' in name
        t = 'c5' in name or t
        t = 'btc' in name or t
        return t

    def maybe_bolster_balance_by_mining(self):
        """ Mine just a little to avoid stream dying due to bankruptcy """
        if self.mine:
            balance = self.get_balance()
            if balance < 0:
                muid_time = time.time()
                key = self.bolster_balance_by_mining(seconds=max(5, int(abs(balance) / 10)))
                mining_time = time.time() - muid_time
                self.mining_time += mining_time
                if key:
                    print('************************')
                    print('     FOUND MUID !!!     ')
                    print('************************', flush=True)
                    self.mining_success += 1
                else:
                    print('Did not find MUID this time', flush=True)

    def downtime(self, seconds, **ignored):
        print(f"Entering downtime for {seconds} seconds")
        #self.maybe_bolster_balance_by_mining() 
        print(f"Done")

    def sample(self, lagged_values, lagged_times=None, name=None, delay=None, **ignored):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        uniques = np.unique(lagged_values)
        chrono_values = list(reversed(lagged_values))
        chrono_times = list(reversed(lagged_times))
        sample_size = self.num_predictions
        df = lagged_to_frame(chrono_times, chrono_values)
        print(df)
        df.plot(x='ds',y='y', 
            title=f"{delay} {name}",
            label='Lagged Values')
        plt.savefig(plotname('lagged.png'))

#        if len(uniques) < 0.3 * len(lagged_values):
#            # arbitrary cutoff of 30% to determine whether outcomes are continuous or quantized
#            v = [x for x in (np.random.choice(lagged_values, self.num_predictions))]
#            print(f"+++ Returning {sample_size} random lagged values.")
#        else:
        if True:
            print(f"--- You have {delay} seconds to return something.")
            before = dt.now()

            # construct a fresh model
            model = Prophet(
                yearly_seasonality=False,
                daily_seasonality=False,
                weekly_seasonality=False,
                ).add_seasonality(name='daily',
                    period=1,
                    fourier_order=15)

            # Fit the model using the lagged data.
            model.fit(df)
            #TODO: determine the frequency dynamically
            freq = '5min'
            futures = model.make_future_dataframe(
                    periods=sample_size,
                    freq=freq,
                    )
            # make the predictions
            pred = model.predict(futures)
            #model.plot(pred)
            pred.plot(x='ds', y='yhat',
                title=f"{delay} {name}",
                label='Predicted')
            plt.savefig(plotname('predicted.png'))

            lapsed = dt.now() - before
            print(f"... {lapsed} seconds later.")
            print(f"+++ Returning {sample_size} predictions")
            print(pred[-sample_size:])
            v = pred[-sample_size:]['yhat']
            return v.values