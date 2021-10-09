from datetime import date
from time import strptime

import numpy as np
import pandas as pd


def process():
    df = pd.read_csv("data/EURUSD.csv")
    df.rename(columns={'date': 'Date', 'value': 'Rate'}, inplace=True)

    for v in ['BOP', 'GDP', 'GGDEBT', 'PPP', 'TERMTRADE', 'INTRATE', 'INFRATE']:
        df_ = pd.read_csv("data/" + v + ".csv")
        df_US = df_.loc[df_['LOCATION'] == df_['LOCATION'].unique()[0]]
        df_EU = df_.loc[df_['LOCATION'] == df_['LOCATION'].unique()[1]].reset_index()
        df[v] = np.nan
        for index, row in df_US.iterrows():
            start_year = str(date(int(row['TIME']), 1, 1))
            end_year = str(date(int(row['TIME']) + 1, 1, 1))
            mask = (df['Date'] >= start_year) & (df['Date'] < end_year)
            if row['Value'] == 0:
                df.loc[mask, v] = 0
            else:
                df.loc[mask, v] = df_EU.iloc[index]['Value'] / row['Value']

    df_US = pd.read_csv("data/US_PPI.csv")
    df_EU = pd.read_csv("data/EU_PPI.csv")
    df['PPI'] = np.nan
    for index, row in df_EU.iterrows():
        start_year = str(strptime(row['TIME'], '%Y-%m'))
        year = int(str(row['TIME'])[0:str(row['TIME']).index("-")])
        year_ = year + 1
        end_year = str(strptime(str(year_) + str(row['TIME'])[str(row['TIME']).index("-")::], '%Y-%m'))
        mask = (df['Date'] >= start_year) & (df['Date'] < end_year)
        df.loc[mask, 'PPI'] = row['Value'] / df_US.iloc[index]['Value']

    df.to_csv("data/FXNET.csv", index=False)


if __name__ == "__main__":
    process()
