# AFP-hedgefund-holdings

## data
begdate = '01/01/2015'
enddate = '12/30/2022'

Universe: all the stocks between the begin and end dates that were included in S&P500, in total of 770 names

### price, volume, GICS

As in daily

- permno
- date
- prc: close price
- vol: volume
- openprc: open price
- askhi: high price
- bidlo: low price
- ret
- shrout: shares outstanding
- Yyyy-mm
- gics: in code format
- comnam: company name
- ticker

### holdings
As in quaterly
- rdate: report date
- mgrno: hedgefund code
- permno: stock code
- shares: shares held by this hedgefund on this stock
- shares_adj: shares*cfacshr
- phrdate: previous report date
- trade: change in current shares and previous shares
- modtrade: modified trade
- qtrgap: quarter gap
- lpermno: lag permno for determining first permno
- npermno: lead permno for determining las permno
- buysale: trade types. 1 = Initial buys; 2 = Incremental (Regular) Buys; -1 = Terminating Sales; -2 = Regular Sales
### hf_netflow
As in quarterly
- assets: total portfolio assets
- pret: forward portfolio return
- tbuys: dollar amount buys per stock by a manager in a quarter
- tsales: dollar amount sells per stock by a manager in a quarter
- tgain: net gain from trades per stock 
- tgainret: Trade Returns = Returns on Purchases - Forgone Returns on Sales
- netflows: net flows in assets
- turnover1: first way to measure turnover Carhart (1997) Turnover Definition
- turnover2: second way to measure turnover by adding Back Net Flows and Redemptions
- turnover3: third way to measure turnover
