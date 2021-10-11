import numpy as np
import pyAgrum as gum

import fxnet


def bayesian_network_inference():
    bn = fxnet.bayesian_network()
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


def decision_network_inference():
    dn = fxnet.decision_network()
    ie = gum.ShaferShenoyLIMIDInference(dn)

    # ie.addEvidence('InterestRate', [1, 0, 0])
    # ie.addEvidence('InflationRate', [1, 0])

    ie.makeInference()
    var = ie.posteriorUtility('Trade').variable('Trade')

    decision_index = np.argmax(ie.posteriorUtility('Trade').toarray())
    decision = var.label(int(decision_index))
    print('EUR/USD Trade Decision: {0}'.format(decision))

    return format(decision)
