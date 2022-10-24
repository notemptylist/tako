import sys
import argparse
from microprediction import new_key, MicroCrawler

def generate_key(difficulty=9):
    """
        Generate new key and return the hash and name
    """
    print(f"Generating key with difficulty ({difficulty})")

    while True:
        nk = new_key(difficulty=difficulty)
        shash, animal = MicroCrawler.shash(nk), MicroCrawler.animal_from_key(nk)
        body = f"key: {nk}\nnom de plum: {animal}\nshash: {shash}"
        print(f"Found key {animal}\n{body}")
    return (nk, MicroCrawler.shash(nk), MicroCrawler.animal_from_key(nk))

def main(args): 
    if args.difficulty:
        print(generate_key(args.difficulty))
        return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--new-key", action="store", type=int, dest='difficulty')

    args = parser.parse_args()
    sys.exit(main(args))