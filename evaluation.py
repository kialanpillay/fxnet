from datetime import date

import numpy as np
import pyAgrum as gum

import networks
from engine import hard_evidence


def bayesian_network_inference():
    bn = networks.bayesian_network()
    ie = gum.LazyPropagation(bn)

    # P(ClosePrice)
    ie.makeInference()
    print(ie.posterior("ClosePrice"))

    # P(ClosePrice | InterestRate = Positive)
    ie.setEvidence({'InterestRate': 'Positive'})
    ie.makeInference()
    print(ie.posterior("ClosePrice"))

    # P(ClosePrice | CurrentAccount = Positive)
    ie.setEvidence({'CurrentAccount': 'Positive'})
    ie.makeInference()
    print(ie.posterior("ClosePrice"))

    # P(ClosePrice | PPI = Negative)
    ie.setEvidence({'PPI': 'Negative'})
    ie.makeInference()
    print(ie.posterior("ClosePrice"))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative)
    ie.setEvidence({'PPI': 'Negative', 'PublicDebt': 'Negative'})
    ie.makeInference()
    print(ie.posterior("ClosePrice"))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative, GDP = Positive)
    ie.setEvidence({'PPI': 'Negative', 'PublicDebt': 'Negative', 'GDP': 'Positive'})
    ie.makeInference()
    print(ie.posterior("ClosePrice"))

    # P(ClosePrice | PPI = Negative, PublicDebt = Negative, GDP = Positive, Sentiment = Negative)
    ie.setEvidence({'PPI': 'Negative', 'PublicDebt': 'Negative', 'GDP': 'Positive', 'Sentiment': 'Negative'})
    ie.makeInference()
    print(ie.posterior("ClosePrice"))


def decision_network_inference(df=None, evidence=False, year=None):
    dn = networks.decision_network()
    ie = gum.ShaferShenoyLIMIDInference(dn)

    if year:
        mask = (df['Date'] >= str(date(year - 1, 1, 1))) & (df['Date'] < str(date(year, 1, 1)))
        df = df.loc[mask]

    if evidence:
        ie.addEvidence('GDP', hard_evidence(df['GDP'].values, 'GDP'))
        ie.addEvidence('InterestRate', hard_evidence(df['INTRATE'].values, 'InterestRate'))
        ie.addEvidence('PPI', hard_evidence(df['PPI'].values, 'PPI'))
        ie.addEvidence('PublicDebt', hard_evidence(df['GGDEBT'].values, 'PublicDebt'))
        ie.addEvidence('InflationRate', hard_evidence(df['INFRATE'].values, 'InflationRate'))
        ie.addEvidence('CurrentAccount', hard_evidence(df['BOP'].values, 'CurrentAccount'))
        ie.addEvidence('TermsOfTrade', hard_evidence(df['TERMTRADE'].values, 'TermsOfTrade'))

    ie.makeInference()
    var = ie.posteriorUtility('Trade').variable('Trade')

    decision_index = np.argmax(ie.posteriorUtility('Trade').toarray())
    decision = var.label(int(decision_index))
    print(ie.posteriorUtility('Trade'))
    print('EUR/USD Trade Decision: {0}'.format(decision))

    return decision
