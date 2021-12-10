#!/usr/bin/env python3
"""
Crawl Streams on microprediction.org
"""
import sys
import argparse
from microprediction import MicroCrawler
from private_config import KEYS
from series import SeriesCrawler
__author__ = 'regulate@gmail.com'
__version__ = '0.1.0'

class SponsorCrawler(MicroCrawler):
    """
    Crawl streams from a specific sponsor.
    """
    def __init__(self, **kwargs):
        self._sponsored_streams = []
        super().__init__(**kwargs)

    def sync_sponsored_streams(self, sponsors):
        """
        Populate a local list of stream names with the specified sponsor.
        """
        self.sponsor = sponsors
        self._sponsored_streams.extend([x[0] for x in self.get_sponsors().items() if x[1] in self.sponsor])
        print(f'Sponsored streams: {self._sponsored_streams}')
        return self._sponsored_streams

    def include_stream(self, name=None, **ignore):
        """
        Filter by sponsor.
        """
        return name in self._sponsored_streams

def crawl_sponsored(sponsors, write_key):
    mc = SponsorCrawler(write_key=write_key)
    mc.sync_sponsored_streams(sponsors)
    mc.set_email(__author__)
    mc.run()

def crawl_generic(write_key):
    mc = MicroCrawler(write_key=write_key)
    mc.set_email(__author__)
    mc.run()

def series_crawler(write_key):
    sc = SeriesCrawler(write_key=write_key)
    sc.set_email(__author__)
    sc.run()

def main(args):
    print(args)
    try:
        key = KEYS[args.key]
    except KeyError:
        print("Specified key id is invalid")
        sys.exit(1)

    if args.series and args.key:
        series_crawler(key)
    if args.sponsors and args.key:
        crawl_sponsored(args.sponsors, key)

    if args.generic:
        crawl_generic(key)
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument("-s", "--sponsors", nargs="+", dest="sponsors", help="Crawl sponsored streams.")
    parser.add_argument("-g", "--generic", action="store_true", dest="generic", help="Crawl generically.")
    parser.add_argument("-p", "--prophet", action="store_true", dest="prophet", help="Crawl using fbprophet.")
    parser.add_argument("-t", "--series", action="store_true", dest="series", help="Crawl using TSA libraries.")
    parser.add_argument("-k", "--key", action="store", dest="key", help=f"Crawl using key ID. Available keys: {list(KEYS.keys())})")
    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    sys.exit(main(args))
