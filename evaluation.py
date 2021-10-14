from datetime import date

import networks
import preprocessing
from inference import bayesian_network_inference, decision_network_inference


def bayesian_network_test():
    # P(ClosePrice)
    bn = networks.bayesian_network()
    bayesian_network_inference(bn, "ClosePrice")
    print(bayesian_network_inference(bn, "ClosePrice"))

    # P(ClosePrice | InterestRate = Positive)
    print(bayesian_network_inference(bn, "ClosePrice", {'InterestRate': 'Positive'}))

    # P(ClosePrice | PPI = Negative)
    print(bayesian_network_inference(bn, "ClosePrice", {'PPI': 'Negative'}))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative, GDP = Positive)
    print(
        bayesian_network_inference(bn, "ClosePrice", {'PPI': 'Negative', 'PublicDebt': 'Negative', 'GDP': 'Positive'}))

    # P(InflationRate | PPI = Negative)
    print(bayesian_network_inference(bn, "InflationRate", {'PPI': 'Negative'}))

    # P(InterestRate | USPoliticalState = Stable)
    print(bayesian_network_inference(bn, "InterestRate", {'USPoliticalState': 'Stable'}))


def backtest(trade_size=1):
    df = preprocessing.load()
    for year in range(2017, 2021):
        mask = (df['Date'] >= str(date(year - 1, 1, 1))) & (df['Date'] < str(date(year, 1, 1)))
        df_prev = df.loc[mask]
        mask = (df['Date'] >= str(date(year, 1, 1))) & (df['Date'] < str(date(year + 1, 1, 1)))
        df_next = df.loc[mask]
        decision = decision_network_inference(evidence=True, df=df, year=year)

        if decision == "Buy":
            multiplier = 1
        elif decision == "Sell":
            multiplier = -1
        else:
            multiplier = 0

        for month in [1, 3, 6]:
            rate_ = df_prev['Rate'].values[-1]
            rate = df_next['Rate'].values[month * 20]

            points = multiplier * (rate - rate_) * 100000
            PL = round(points * trade_size, 2)
            print('{0}-Month Profit/Loss ({1}): ${2}'.format(month, year, PL))
        print()


def use_case():
    # Retail Investor
    print("Use Case 1: Retail Investor")
    prior_information = {'InterestRate': 'Positive', 'GDP': 'Positive'}
    decision_network_inference(evidence=True, prior_information=prior_information)
    print()

    # Company
    print("Use Case 1: Company")
    prior_information = {'PPI': 'Positive'}
    decision_network_inference(evidence=True, prior_information=prior_information)
    print()

    # Fund Manager
    print("Use Case 1: Fund Manager")
    prior_information = {'CurrentAccount': 'Negative', 'GDP': 'Negative', 'InflationRate': 'Negative'}
    decision_network_inference(evidence=True, prior_information=prior_information)
    print()
