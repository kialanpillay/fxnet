from datetime import date

import preprocessing
from inference import bayesian_network_inference, decision_network_inference


def bayesian_network_test():
    # P(ClosePrice)
    bayesian_network_inference("ClosePrice")
    print(bayesian_network_inference("ClosePrice"))

    # P(ClosePrice | InterestRate = Positive)
    print(bayesian_network_inference("ClosePrice", {'InterestRate': 'Positive'}))

    # P(ClosePrice | CurrentAccount = Positive)
    print(bayesian_network_inference("ClosePrice", {'CurrentAccount': 'Positive'}))

    # P(ClosePrice | PPI = Negative)
    print(bayesian_network_inference("ClosePrice", {'PPI': 'Negative'}))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative)
    print(bayesian_network_inference("ClostPrice", {'PPI': 'Negative', 'PublicDebt': 'Negative'}))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative, GDP = Positive)
    print(bayesian_network_inference("ClosePrice", {'PPI': 'Negative', 'PublicDebt': 'Negative', 'GDP': 'Positive'}))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative, GDP = Positive, Sentiment = Negative)
    print(bayesian_network_inference("ClosePrice", {'PPI': 'Negative', 'PublicDebt': 'Negative', 'GDP': 'Positive',
                                                    'Sentiment': 'Negative'}))

    # P(InflationRate | PPI = Negative)
    print(bayesian_network_inference("InflationRate", {'PPI': 'Negative'}))

    # P(InterestRate | USPoliticalState = Stable)
    print(bayesian_network_inference("InterestRate", {'USPoliticalState': 'Stable'}))


def backtest(trade_size=1):
    df = preprocessing.load()
    for year in range(2017, 2021):
        mask = (df['Date'] >= str(date(year - 1, 1, 1))) & (df['Date'] < str(date(year, 1, 1)))
        df_prev = df.loc[mask]
        mask = (df['Date'] >= str(date(year, 1, 1))) & (df['Date'] < str(date(year + 1, 1, 1)))
        df_next = df.loc[mask]
        decision = decision_network_inference(df, evidence=True, year=year)

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
            print('EUR/USD {0}-Month Profit/Loss ({1}): ${2}'.format(month, year, PL))
        print()