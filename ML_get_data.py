import numpy as np
import pandas as pd
import pyarrow.parquet as pq


### Factors Data
parquet_file_path = '../data/factor_all.parquet.gzip'
factors_monthly = pq.read_table(parquet_file_path)
factors_monthly = factors_monthly.to_pandas()
factors_monthly['permno'] = factors_monthly['permno'].apply(int)
factors_quarterly = factors_monthly.groupby(['permno', pd.Grouper(key='date', freq='Q')]).mean()
factors_quarterly.reset_index(inplace=True)


# Holdings Data (from fund_holding_signals.ipynb)
parquet_file_path = '../data/holdings_all_hf.parquet.gzip'
holdings = pq.read_table(parquet_file_path)
holdings = holdings.to_pandas()
holdings = holdings.reset_index().set_index(['date', 'mgrno', 'permno']).sort_index()
holdings.loc[holdings.index.get_level_values(0) == holdings.index.get_level_values(0).min(), 'buysale'] = 0  # the start of data cannot be treated as initial buy
holdings['buysale'] = np.sign(holdings['buysale'])
holdings['value'] = holdings['shares']*holdings['prc']
holdings['portfolio_value'] = holdings.groupby(['date', 'mgrno'])['value'].transform('sum')
holdings['weight'] = holdings['value'] / holdings['portfolio_value'] - holdings['mkt_port_weight']
holdings['weight_diff'] = holdings.groupby(['mgrno', 'permno'])['weight'].diff()
holdings['weight_diff'] = np.where(holdings['buysale']==1, holdings['weight'], holdings['weight_diff'])
holdings['trade_dollar'] = holdings['trade'] * holdings['prc']
holdings['trade_dollar_pct'] = holdings['trade_dollar'] / holdings['portfolio_value']
holdings['share_chg'] = holdings['trade'] / (holdings['shares'] - holdings['trade'])
holdings['value_chg'] = holdings.groupby(['mgrno', 'permno'])['value'].pct_change()
mgrno_cnt = holdings.reset_index().set_index(['date', 'permno']).groupby(['date', 'permno'])['mgrno'].count()
mgrno_cnt.name = 'mgrno_cnt'
total_mgrno_cnt = holdings.reset_index().groupby(['date'])['mgrno'].count()
total_mgrno_cnt.name = 'total_mgrno_cnt'
mgrno_cnt = total_mgrno_cnt.to_frame().join(mgrno_cnt)
mgrno_cnt['mgrno_cnt_ratio'] = mgrno_cnt['mgrno_cnt'] / mgrno_cnt['total_mgrno_cnt']
mgrno_cnt['mgrno_cnt_ratio_chg'] = mgrno_cnt.groupby('permno')['mgrno_cnt_ratio'].pct_change()
mgrno_cnt['mgrno_cnt_ratio_chg'].replace([np.inf, -np.inf], 0, inplace=True)
total_value = holdings.groupby(['date', 'permno'])['value'].sum()
total_value.name = 'total_value'
total_value = total_value.to_frame()
total_value['value_chg'] = total_value.groupby('permno')['total_value'].pct_change()
total_value['value_chg'] = np.where(total_value['total_value']==0, 0, total_value['value_chg'])
total_value['value_chg'].replace([np.inf, -np.inf], 0, inplace=True)

# calculate signal average among all mgrs
holdings_stk = holdings.groupby(['date', 'permno']).agg({'weight':'mean', 'weight_diff': 'mean', 'buysale': 'mean', 'ret': 'last'})
holdings_stk['weight_chg'] = holdings_stk.groupby('permno')['weight'].pct_change()
holdings_stk['weight_chg'].replace([np.inf, -np.inf], 0, inplace=True)
holdings_stk = holdings_stk.join(mgrno_cnt[['mgrno_cnt_ratio', 'mgrno_cnt_ratio_chg']])
holdings_stk = holdings_stk.join(total_value)
holdings_stk = holdings_stk.reset_index()
holdings_stk = holdings_stk[['date', 'permno', 'weight', 'weight_diff', 'ret']]



# Data to be used in machine learning
data = pd.merge(holdings_stk, factors_quarterly, on=['permno', 'date'], how='left')
data.to_csv('../data/ML_data_raw.csv')
data.dropna().to_csv('../data/ML_data_clean.csv')