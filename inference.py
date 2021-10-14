from datetime import date

import numpy as np
import pyAgrum as gum

import networks


def bayesian_network_inference(bn, target, evidence={}):
    ie = gum.VariableElimination(bn)

    # P(target | evidence)
    ie.setEvidence(evidence)
    ie.makeInference()
    return ie.posterior(target)


def decision_network_inference(evidence=False, df=None, prior_information={}, year=None):
    dn = networks.decision_network()
    ie = gum.ShaferShenoyLIMIDInference(dn)

    if year:
        mask = (df['Date'] >= str(date(year - 1, 1, 1))) & (df['Date'] < str(date(year, 1, 1)))
        df = df.loc[mask]

    if evidence and df is not None:
        ie.addEvidence('GDP', hard_evidence(df['GDP'].values, 'GDP'))
        ie.addEvidence('InterestRate', hard_evidence(df['INTRATE'].values, 'InterestRate'))
        ie.addEvidence('PPI', hard_evidence(df['PPI'].values, 'PPI'))
        ie.addEvidence('PublicDebt', hard_evidence(df['GGDEBT'].values, 'PublicDebt'))
        ie.addEvidence('InflationRate', hard_evidence(df['INFRATE'].values, 'InflationRate'))
        ie.addEvidence('CurrentAccount', hard_evidence(df['BOP'].values, 'CurrentAccount'))
        ie.addEvidence('TermsOfTrade', hard_evidence(df['TERMTRADE'].values, 'TermsOfTrade'))

    if evidence and prior_information:
        for name, state in prior_information.items():
            ie.addEvidence(name, convert_evidence(state, name))

    ie.makeInference()
    var = ie.posteriorUtility('Trade').variable('Trade')

    decision_index = np.argmax(ie.posteriorUtility('Trade').toarray())
    decision = var.label(int(decision_index))
    # print(ie.posteriorUtility('Trade'))
    print('EUR/USD Trade Decision: {0}'.format(decision))

    return decision


def hard_evidence(feature, name):
    e = feature[-1]
    if name == 'InterestRate':
        if e > 0:
            return [1, 0, 0]
        elif e == 0:
            return [0, 1, 0]
        else:
            return [0, 0, 1]
    elif name in ['InflationRate', 'GDP', 'PPI', 'PublicDebt', 'CurrentAccount', 'TermsOfTrade']:
        if e > 0:
            return [1, 0]
        else:
            return [0, 1]
    elif name == 'ClosePrice':
        e_ = feature[-60]
        if e / e_ > 1.01:
            return [1, 0, 0]
        elif e / e_ < 0.99:
            return [0, 0, 1]
        else:
            return [0, 1, 0]
    else:
        pass


def convert_evidence(state, name):
    s = state
    if name == 'InterestRate':
        if s == 'Positive':
            return [1, 0, 0]
        elif s == 'Equal':
            return [0, 1, 0]
        else:
            return [0, 0, 1]
    elif name in ['InflationRate', 'GDP', 'PPI', 'PublicDebt', 'CurrentAccount', 'TermsOfTrade']:
        if s == 'Positive':
            return [1, 0]
        else:
            return [0, 1]
    elif name == 'ClosePrice':
        if s == 'Up':
            return [1, 0, 0]
        elif s == 'Down':
            return [0, 0, 1]
        else:
            return [0, 1, 0]
    else:
        pass
