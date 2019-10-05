# EasyBT

Small library for backtesting quantitative strategies

Python 3, Pandas, NumPy, Plotly for plotting

## Installation

Install by pip:

```
pip install git+https://github.com/Fsa55/easybt
```

Or directly:

```
git clone https://github.com/Fsa55/easybt.git
cd easybt
python setup.py install
```


## Usage

```python
from easybt import Backtest
```
The idea is to inherit Backtest class and override preprocess method (and strategy method if needed), where we should create trading signals.

Lets  create Bollinger Bands strategy in which we 
* Enter long position when close price crosses up lower band 
* Enter short position when close price crosses down upper band
* Exit long when close price crosses up SMA
* Exit short when close price crosses down SMA

```python
class Bollinger_Bands_Strategy(Backtest):
    def preprocess(self):
        # creating Bollinger Bands with period = 50
        self.data['SMA50'] = self.data.close.rolling(window = 50).mean()
        self.data['STD50'] = self.data.close.rolling(window = 50).std()
        self.data['TOP_BOL2'] = self.data['SMA50'] + 2*self.data['STD50']
        self.data['BOT_BOL2'] = self.data['SMA50'] - 2*self.data['STD50']
        # Buy when close price crosses up bottom Bollinger Band 
        self.data['buy_signal'] = np.where((self.data.close > self.data.BOT_BOL2)&(self.data.close.shift() < self.data.BOT_BOL2.shift()),1,0)
        # Sell when close price crosses down upper Bollinger Band
        self.data['sell_signal'] = np.where((self.data.close < self.data.TOP_BOL2)&(self.data.close.shift() > self.data.TOP_BOL2.shift()),1,0)
        # Exit long position when close crosses up SMA50
        self.data['exit_buy_signal'] = np.where((self.data.close > self.data.SMA50)&(self.data.close.shift() < self.data.SMA50.shift()),1,0)
        # Exit short position when close crosses down SMA50
        self.data['exit_sell_signal'] = np.where((self.data.close < self.data.SMA50)&(self.data.close.shift() > self.data.SMA50.shift()),1,0)
```
Create class instance and run the backtest.
```python
AAPL_Bol_Bands  = Bollinger_Bands_Strategy(AAPL_test_data)
AAPL_Bol_Bands.run()
```
Input data format is OHLC dataframe with DateTime index. Additionaly you can pass columns with needed indicators/calculations for trading signals creation. If you have just close price: copy it to open, high and low columns. 

| open   |   high |   low | close |
|:----------|--------------:|----------------:|----------|
| 27.42    |  27.56 |   27.29 | 27.50

Some numbers to evaluate our result.
```python
AAPL_Bol_Bands.stats
```
```
{'Num of closed trades': 32,
 'Num of Win trades': 24,
 'Num of Loss trades': 8,
 'Win/Loss ratio': 3.0,
 'Win rate [%]': 0.75,
 'Sharp ratio': 0.02,
 'Max drowdown [%]': -1.92}
```
Trades list.
```python
AAPL_Bol_Bands.trades_df.head(3)
```
```
2010-04-27	33.6758	SELL
2010-05-07	30.3113	EXIT_SELL
2010-06-21	34.7206	SELL
```
Plot the results
```python
AAPL_Bol_Bands.plot(render = None,sub_ind=[['STD50']])
# render = 'browser' to open chart in new default browser window
```
![]([![](https://raw.githubusercontent.com/Fsa55/easybt/master/backtest.png)](https://raw.githubusercontent.com/Fsa55/easybt/master/backtest.png))
