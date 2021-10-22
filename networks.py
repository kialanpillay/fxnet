import pyAgrum as gum


def bayesian_network():
    bn = gum.BayesNet('FXNet')

    node = gum.LabelizedVariable('InterestRate', '', 3)
    node.changeLabel(0, 'Positive')
    node.changeLabel(1, 'Equal')
    node.changeLabel(2, 'Negative')
    bn.add(node)

    for label in ['InflationRate', 'GDP', 'PPI', 'PublicDebt', 'CurrentAccount', 'TermsOfTrade', 'USDSentiment']:
        node = gum.LabelizedVariable(label, '', 2)
        node.changeLabel(0, 'Positive')
        node.changeLabel(1, 'Negative')
        bn.add(node)

    node = gum.LabelizedVariable('ClosePrice', '', 3)
    node.changeLabel(0, 'Up')
    node.changeLabel(1, 'Sideways')
    node.changeLabel(2, 'Down')
    bn.add(node)

    node = gum.LabelizedVariable('USPoliticalState', '', 2)
    node.changeLabel(0, 'Stable')
    node.changeLabel(1, 'Unstable')
    bn.add(node)

    bn.addArc(bn.idFromName('USPoliticalState'), bn.idFromName('USDSentiment'))
    bn.addArc(bn.idFromName('USDSentiment'), bn.idFromName('ClosePrice'))

    bn.addArc(bn.idFromName('USPoliticalState'), bn.idFromName('GDP'))
    bn.addArc(bn.idFromName('GDP'), bn.idFromName('InterestRate'))
    bn.addArc(bn.idFromName('InterestRate'), bn.idFromName('ClosePrice'))

    bn.addArc(bn.idFromName('CurrentAccount'), bn.idFromName('TermsOfTrade'))
    bn.addArc(bn.idFromName('TermsOfTrade'), bn.idFromName('ClosePrice'))

    bn.addArc(bn.idFromName('PublicDebt'), bn.idFromName('InflationRate'))
    bn.addArc(bn.idFromName('PPI'), bn.idFromName('InflationRate'))
    bn.addArc(bn.idFromName('InflationRate'), bn.idFromName('ClosePrice'))

    bn.cpt("USPoliticalState").fillWith([0.9, 0.1])

    bn.cpt("GDP")[{'USPoliticalState': 'Stable'}] = [0.4, 0.6]
    bn.cpt("GDP")[{'USPoliticalState': 'Unstable'}] = [0.7, 0.3]

    bn.cpt("USDSentiment")[{'USPoliticalState': 'Stable'}] = [0.6, 0.4]
    bn.cpt("USDSentiment")[{'USPoliticalState': 'Unstable'}] = [0.1, 0.9]

    bn.cpt("InterestRate")[{'GDP': 'Positive'}] = [0.7, 0.25, 0.05]
    bn.cpt("InterestRate")[{'GDP': 'Negative'}] = [0.05, 0.25, 0.7]

    bn.cpt("CurrentAccount").fillWith([0.5, 0.5])
    bn.cpt("TermsOfTrade")[{'CurrentAccount': 'Positive'}] = [0.8, 0.2]
    bn.cpt("TermsOfTrade")[{'CurrentAccount': 'Negative'}] = [0.2, 0.8]

    bn.cpt("PublicDebt").fillWith([0.5, 0.5])
    bn.cpt("PPI").fillWith([0.5, 0.5])

    bn.cpt("InflationRate")[{'PublicDebt': 'Positive', 'PPI': 'Positive'}] = [0.9, 0.1]
    bn.cpt("InflationRate")[{'PublicDebt': 'Negative', 'PPI': 'Positive'}] = [0.6, 0.4]
    bn.cpt("InflationRate")[{'PublicDebt': 'Positive', 'PPI': 'Negative'}] = [0.4, 0.6]
    bn.cpt("InflationRate")[{'PublicDebt': 'Negative', 'PPI': 'Negative'}] = [0.1, 0.9]

    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.5, 0.3, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.2, 0.5]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.3, 0.4, 0.3]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.2, 0.3, 0.5]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.05, 0.05, 0.9]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.2, 0.2, 0.6]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.6, 0.2, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.05, 0.05, 0.9]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.15, 0.15, 0.7]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.1, 0.1, 0.8]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.3, 0.4, 0.3]

    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.9, 0.05, 0.05]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.6, 0.2, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.5, 0.3, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.8, 0.2, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.3, 0.6]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.2, 0.3, 0.5]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.2, 0.4, 0.4]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.6, 0.2, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.5, 0.3, 0.2]
    bn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.6, 0.2, 0.2]

    gum.saveBN(bn, 'output/fx_bayesian_network.bif')
    return bn


def decision_network():
    dn = gum.InfluenceDiagram()

    node = gum.LabelizedVariable('InterestRate', '', 3)
    node.changeLabel(0, 'Positive')
    node.changeLabel(1, 'Equal')
    node.changeLabel(2, 'Negative')
    dn.addChanceNode(node)

    for label in ['InflationRate', 'GDP', 'PPI', 'PublicDebt', 'CurrentAccount', 'TermsOfTrade', 'USDSentiment']:
        node = gum.LabelizedVariable(label, '', 2)
        node.changeLabel(0, 'Positive')
        node.changeLabel(1, 'Negative')
        dn.addChanceNode(node)

    node = gum.LabelizedVariable('ClosePrice', '', 3)
    node.changeLabel(0, 'Up')
    node.changeLabel(1, 'Sideways')
    node.changeLabel(2, 'Down')
    dn.addChanceNode(node)

    node = gum.LabelizedVariable('USPoliticalState', '', 2)
    node.changeLabel(0, 'Stable')
    node.changeLabel(1, 'Unstable')
    dn.addChanceNode(node)

    dn.addArc(dn.idFromName('USPoliticalState'), dn.idFromName('USDSentiment'))
    dn.addArc(dn.idFromName('USDSentiment'), dn.idFromName('ClosePrice'))

    dn.addArc(dn.idFromName('USPoliticalState'), dn.idFromName('GDP'))
    dn.addArc(dn.idFromName('GDP'), dn.idFromName('InterestRate'))
    dn.addArc(dn.idFromName('InterestRate'), dn.idFromName('ClosePrice'))

    dn.addArc(dn.idFromName('CurrentAccount'), dn.idFromName('TermsOfTrade'))
    dn.addArc(dn.idFromName('TermsOfTrade'), dn.idFromName('ClosePrice'))

    dn.addArc(dn.idFromName('PublicDebt'), dn.idFromName('InflationRate'))
    dn.addArc(dn.idFromName('PPI'), dn.idFromName('InflationRate'))
    dn.addArc(dn.idFromName('InflationRate'), dn.idFromName('ClosePrice'))

    dn.cpt("USPoliticalState").fillWith([0.9, 0.1])

    dn.cpt("GDP")[{'USPoliticalState': 'Stable'}] = [0.4, 0.6]
    dn.cpt("GDP")[{'USPoliticalState': 'Unstable'}] = [0.7, 0.3]

    dn.cpt("USDSentiment")[{'USPoliticalState': 'Stable'}] = [0.6, 0.4]
    dn.cpt("USDSentiment")[{'USPoliticalState': 'Unstable'}] = [0.1, 0.9]

    dn.cpt("InterestRate")[{'GDP': 'Positive'}] = [0.7, 0.25, 0.05]
    dn.cpt("InterestRate")[{'GDP': 'Negative'}] = [0.05, 0.25, 0.7]

    dn.cpt("CurrentAccount").fillWith([0.5, 0.5])
    dn.cpt("TermsOfTrade")[{'CurrentAccount': 'Positive'}] = [0.8, 0.2]
    dn.cpt("TermsOfTrade")[{'CurrentAccount': 'Negative'}] = [0.2, 0.8]

    dn.cpt("PublicDebt").fillWith([0.5, 0.5])
    dn.cpt("PPI").fillWith([0.5, 0.5])

    dn.cpt("InflationRate")[{'PublicDebt': 'Positive', 'PPI': 'Positive'}] = [0.9, 0.1]
    dn.cpt("InflationRate")[{'PublicDebt': 'Negative', 'PPI': 'Positive'}] = [0.6, 0.4]
    dn.cpt("InflationRate")[{'PublicDebt': 'Positive', 'PPI': 'Negative'}] = [0.4, 0.6]
    dn.cpt("InflationRate")[{'PublicDebt': 'Negative', 'PPI': 'Negative'}] = [0.1, 0.9]

    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.5, 0.3, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.2, 0.5]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.3, 0.4, 0.3]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.2, 0.3, 0.5]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.05, 0.05, 0.9]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.2, 0.2, 0.6]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.6, 0.2, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.05, 0.05, 0.9]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.15, 0.15, 0.7]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.1, 0.1, 0.8]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Positive', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.3, 0.4, 0.3]

    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.9, 0.05, 0.05]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.6, 0.2, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.5, 0.3, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.8, 0.2, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.3, 0.6]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Equal', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.2, 0.3, 0.5]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.2, 0.4, 0.4]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Negative'}] = [0.6, 0.2, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Negative', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Positive'}] = [0.3, 0.4, 0.3]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Negative',
                          'InflationRate': 'Negative'}] = [0.5, 0.3, 0.2]
    dn.cpt("ClosePrice")[{'USDSentiment': 'Negative', 'InterestRate': 'Positive', 'TermsOfTrade': 'Positive',
                          'InflationRate': 'Positive'}] = [0.6, 0.2, 0.2]

    U = gum.LabelizedVariable('TradeUtility', '', 1)
    dn.addUtilityNode(U)

    D = gum.LabelizedVariable('Trade', '', 3)
    D.changeLabel(0, 'Buy')
    D.changeLabel(1, 'No Action')
    D.changeLabel(2, 'Sell')
    dn.addDecisionNode(D)

    dn.addArc(dn.idFromName('PPI'), dn.idFromName('Trade'))

    dn.addArc(dn.idFromName('ClosePrice'), dn.idFromName('TradeUtility'))
    dn.addArc(dn.idFromName('Trade'), dn.idFromName('TradeUtility'))

    dn.utility(dn.idFromName('TradeUtility'))[{'ClosePrice': 'Up'}] = [[100], [20], [-100]]
    dn.utility(dn.idFromName('TradeUtility'))[{'ClosePrice': 'Sideways'}] = [[20], [40], [20]]
    dn.utility(dn.idFromName('TradeUtility'))[{'ClosePrice': 'Down'}] = [[-100], [20], [100]]

    gum.saveBN(dn, 'output/fxnet.bifxml')

    return dn
