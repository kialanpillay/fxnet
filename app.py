import argparse

from art import tprint

import evaluation
import inference
import preprocessing


def main():
    tprint("FXNet")
    print("FOREX Trading Decision Support Tool")
    print("© FXNet 2021")
    print("-" * 40)
    print()

    if args.test:
        evaluation.bayesian_network_test()

    if args.predict:
        df = preprocessing.load()
        inference.decision_network_inference(df=df, evidence=args.evidence)
        if args.test:
            evaluation.backtest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EUR/USD Trading Decision Support Tool")
    parser.add_argument('--test', action='store_true', help="Bayesian network inference")
    parser.add_argument('--predict', action='store_true', help="Decision network decision support")
    parser.add_argument('--evidence', action='store_true', help="Generate hard evidence")
    args = parser.parse_args()
    main()
