import json
from datetime import datetime as dt
from proph import lagged_to_frame
from microprediction import MicroCrawler
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from proph import plotname

def load_lagged(name):
    from proph import lagged_to_frame
    data = json.load(open(name, 'r'))
    times, values = list(reversed(data[1])), list(reversed(data[0]))
    df = lagged_to_frame(times, values)
    return df


def arima(data):
    model = ARIMA(data['y'], order=(0, 0, 1))
    model_fit = model.fit()
    yhat = model_fit.predict(len(data), len(data))
    print(yhat)

class SeriesCrawler(MicroCrawler):

    def include_sponsor(self, sponsor=None, **ignore):
        """ Override this as you see fit to select streams for your crawler """
        spons = ['Emblossom Moth', 'Offcast Goose']
        #'e3b1055033076108b4279c473cde3a67', '0ffca579005ef5d8757270f007c4db76']
        return sponsor in spons

    def exclude_stream(self, name=None, **ignore):
        return '~' in name

    def include_stream(self, name=None, **ignore):
        return 'c5' in name

    def sample(self, lagged_values, lagged_times=None, name=None, delay=None, **ignored):
        """ Find Unique Values to see if outcomes are discrete or continuous """
        chrono_values = list(reversed(lagged_values))
        chrono_times = list(reversed(lagged_times))
        sample_size = self.num_predictions
        df, res = lagged_to_frame(chrono_times, chrono_values)
        print(df)
        df.plot(x='ds',y='y', 
            title=f"{delay} {name}",
            label='Lagged Values')
        plt.savefig(plotname('lagged.png'))
        plt.close()
        print(f"--- You have {delay} seconds to return something.")
        before = dt.now()


        model = ARIMA(df['y'].values, order=(1, 1, 1))
        model_fit = model.fit()
        yhats = []
        for x in range(sample_size):
            yhats.extend( model_fit.predict(sample_size, sample_size, typ='levels'))
        print(yhats)
#        plt.savefig(plotname('predicted.png'))
#        plt.close()
#        model.plot_components(pred)
#        plt.savefig(plotname('components.png'))
#        plt.close()

        lapsed = dt.now() - before
        print(f"... {lapsed} seconds later.")
        print(f"+++ Returning {sample_size} predictions")
        return yhats


if __name__ == '__main__':
    arima(load_lagged('c5_etherium.json'))