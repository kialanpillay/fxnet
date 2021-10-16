# FXNet

FOREX Trading Decision Support Tool

<hr>

## Background

The foreign exchange (FX) market is a global decentralised market that facilitates the trading of currencies between
market participants. The FX market is the largest global financial market, with a daily trading volume of approximately
$6.6 trillion in 2019. The modern FX market can trace its inception to the disbandment of the Bretton Woods accords
in 1970. 

The Bretton Woods system governed inter-national monetary management and fixed exchange rates relative to the
US dollar. The critical switch from a fixed rate scheme to a floating exchange rate catalysed the formation of the FX
market. The FX market operates on a 24/5 schedule, and market participants include central banks, commercial banks,hedge
funds, multinational corporations and individual investors (speculators).

Currencies are traded as exchange rate or
currency pairs. A currency pair (Base/Quote) is the quotation of two currencies, forming a ratio of the value of the base
currency over the quote currency, termed exchange rate. Therefore, EUR/USD is the exchange rate for trading the euro
against the United States (US) Dollar. 

The exchange rate, determined by supply and demand and other market factors,
reflects the quote currency amount required to purchase a single unit of the base currency. The EUR/USD pair is the most
liquid (exchanged) currency pair, accounting for approximately a quarter of daily trading volume.

The FX market has high volatility due to multiple erratic economic and geopolitical factors. Therefore, there is inherent complexity
involved in trading currency pairs, especially for speculative purposes. Speculative trades are profit-motivated, seeking
to take advantage of the increase or decrease in the relative value of a currency, often over short time-frames. 

Trading
currencies, therefore, carries a high risk to capital for market participants. This risk disproportionately affects
individual investors, who over-leverage limited capital or incur debt to trade in the FX market. Without an adequate
understanding of the market and relevant economic information, an investor is at risk of rapid insolvency.

## Objective

The FX market is a dynamic and chaotic system that is difficult to accurately model. The existing literature investigating
FX market prediction has focused primarily on Machine and Deep Learning techniques that are black-box and provide minimal
decision support. With black-box approaches, the decision-making rationale is opaque and cannot be explained to the user.
Further, the problem is formulated as exchange rate prediction in the recent studies rather than providing an explicit
Yes/No decision. 

Our proposed Bayesian network model is an explainable or glass-box Artificial Intelligence
technique in which the model inference can be extracted and explained to the user. Our model is focused specifically on
the EUR/USD currency pair, given its liquidity and global demand. The tool will be used to determine whether to buy,
sell, or not trade the EUR/USD currency pair to improve the long-term investment decisions of different classes of market
participants

## User Manual

FXNet Tests
1. Bayesian Network Inference Test
1. Decision Network Profit/Loss Backtest
3. Decision Network Use Case Simulation
```
python3 app.py --test
```

EUR/USD Trading Decision Support
```
python3 app.py --test --predict
```

EUR/USD Trading Decision Support w/ Available Evidence
```
python3 app.py --predict --evidence
```

FXNet Interactive Mode
```
python3 app.py --interactive
```