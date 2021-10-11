import argparse

import pandas as pd


def main():
    df = pd.read_csv("data/FXNET.csv")
    df = df[~df.isnull().any(axis=1)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()
