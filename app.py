import argparse
import json

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

    if args.interactive:
        option = 0
        while option not in [1, 2, 3]:
            print("FXNet Tool Options")
            print("(1) No Evidence")
            print("(2) Available Evidence")
            print("(3) Custom Prior Information")
            option = eval(input("Select Option\n"))

        if option == 1:
            df = preprocessing.load()
            inference.decision_network_inference(evidence=False, df=df)
            print()
        elif option == 2:
            df = preprocessing.load()
            inference.decision_network_inference(evidence=True, df=df)
            print()
        elif option == 3:
            nodes = json.load(open('nodes.json'))
            prior_information = {}

            for N, S in nodes.items():
                option = input("{0} Node Evidence? (Y/N)\n".format(N))
                if option.lower() == 'y' or option.lower() == "yes":
                    for i, s in enumerate(S):
                        print("({0}) {1}".format(i + 1, s))
                    idx = 0
                    while idx not in list(range(1, len(S)+1)):
                        idx = eval(input("Select Option\n"))
                    prior_information[N] = S[idx-1]
                else:
                    continue
            inference.decision_network_inference(evidence=True, prior_information=prior_information)
            print()
    else:
        if args.predict:
            df = preprocessing.load()
            inference.decision_network_inference(evidence=args.evidence, df=df)
            print()

        if args.test:
            evaluation.bayesian_network_test()
            evaluation.backtest()
            evaluation.use_case()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EUR/USD Trading Decision Support Tool")
    parser.add_argument('--test', action='store_true', help="Bayesian network inference")
    parser.add_argument('--predict', action='store_true', help="Decision network decision support")
    parser.add_argument('--evidence', action='store_true', help="Generate hard evidence")
    parser.add_argument('--interactive', action='store_true', help="Interactive mode")
    args = parser.parse_args()
    main()
