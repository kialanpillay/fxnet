def hard_evidence(feature, label):
    e = feature[-1]
    if label == 'InterestRate':
        if e > 0:
            return [1, 0, 0]
        elif e == 0:
            return [0, 1, 0]
        else:
            return [0, 0, 1]
    elif label in ['InflationRate', 'GDP', 'PPI', 'PublicDebt', 'CurrentAccount', 'TermsOfTrade']:
        if e > 0:
            return [1, 0]
        else:
            return [0, 1]
    elif label == 'ClosePrice':
        e_ = feature[-60]
        if e / e_ > 1.01:
            return [1, 0, 0]
        elif e / e_ < 0.99:
            return [0, 0, 1]
        else:
            return [0, 1, 0]
    else:
        pass
