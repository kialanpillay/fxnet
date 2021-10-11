import argparse

import pandas as pd

from evaluation import bayesian_network_inference, decision_network_inference


def main():
    df = pd.read_csv("data/FXNET.csv")
    df = df[~df.isnull().any(axis=1)]

    if args.test:
        bayesian_network_inference()

    decision_network_inference()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true', help="Evaluation mode")
    args = parser.parse_args()
    main()
