from _plotly_future_ import v4_subplots
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


class Statistics:
    # Get rid from dict with statistics, bringing it to this class
    # TODO: Complete

    def __init__(self, closed_trades: int):
        self.closed_trades = closed_trades

    @property
    def closed_trades(self) -> int:
        """
        :return: Num of closed trades
        """
        return self.closed_trades

class Backtest:
    """
    This class perfoms Backtest for 1 lot only (Max position = either 1 or -1 of a stock or future contract)
    self.trades_df: Dataframe with trades
    self.stats: Dict of strategy statistics
    """

    def __init__(self, data, cost_per_trade=0.0, multiplier=1.0):
        """
        Parameters:
        -----------
        data: pandas Dataframe with Datetime index containing ohlc data, indicators or 
        whatever to build trading signals on)
        -----------
        cost_per_trade: cost to pay for one trade. Default 0
        -----------
        multiplier: For stocks it is equal to 1. For futures multiplier can be found in contract's scpecification document
        (example: Light Sweet Crude Oil Future contract (CL, Nymex) multiplier is equal to 1000) default = 1.
        """
        self.data = data.copy()
        self.data = self.data.loc[~self.data.index.duplicated(keep='last')]

        # TODO: docs required!
        self.data['buy'] = None
        self.data['sell'] = None
        self.data['exit_buy'] = None
        self.data['exit_sell'] = None

        self.data['buy_signal'] = None
        self.data['sell_signal'] = None
        self.data['exit_buy_signal'] = None
        self.data['exit_sell_signal'] = None

        self.trades_df = None
        self.position = 0
        self.cost_per_trade = cost_per_trade
        self.multiplier = multiplier

        # TODO: replace with data class
        self.stats = {'Num of closed trades': 0,
                      'Num of Win trades': 0,
                      'Num of Loss trades': 0,
                      'Win/Loss ratio': 0,
                      'Win rate [%]': 0,
                      'Sharp ratio': 0,
                      'Max drowdown [%]': 0
                      }

    def preprocess(self, fata):
        """
        Initialize your trading signals here with zeros and ones.
        self.data['buy_signal'] = *Your logic here*
        self.data['sell_signal'] = *Your logic here*
        self.data['exit_buy_signal'] = *Your logic here*
        self.data['exit_sell_signal'] = *Your logic here*
        -----------
        Example:
        self.data['buy_signal'] = np.where(self.data.SMA100 > self.data.SMA200, 1, 0)
        
        """
        # pass
        raise NotImplementedError

    # TODO: BETTER NAMING
    def strategy(self, index, row):
        """
        Default:
        Buy when self.position == 0 and self.data['buy_signal'] == 1
        Sell when self.position == 0 and self.data['sell_signal'] == 1
        Exit Buy when self.position == 1 and self.data['exit_buy_signal'] == 1
        Exit Sell when self.position == -1 and self.data['exit_sell_signal'] == 1
        """
        # simplify ?
        if self.position == 0:
            if row.sell_signal == 1:
                self.make_trade('sell', index)
                self.position = -1.0
            if row.buy_signal == 1:
                self.make_trade('buy', index)
                self.position = 1.0

        if self.position == -1:
            if row.exit_sell_signal >= 1:
                self.make_trade('exit_sell', index)
                self.position = 0
        elif self.position == 1:
            if row.exit_buy_signal >= 1:
                self.make_trade('exit_buy', index)
                self.position = 0

    # TODO: Docs required
    def make_trade(self, side, index):
        self.data.loc[index, side] = self.data.loc[index, 'close']


    

    def _create_trades_list(self):
        tradesbuy = pd.DataFrame(data=self.data.buy.dropna().values, index=self.data.buy.dropna().index,
                                 columns=['price'])

        tradesbuy['type'] = 'BUY' # TODO: MOVE TO CONSTS OR ENUM !!
        tradessell = pd.DataFrame(data=self.data.sell.dropna().values, index=self.data.sell.dropna().index,
                                  columns=['price'])

        tradessell['type'] = 'SELL'
        tradesexit_sell = pd.DataFrame(data=self.data.exit_sell.dropna().values,
                                       index=self.data.exit_sell.dropna().index, columns=['price'])

        tradesexit_sell['type'] = 'EXIT_SELL'
        tradesexit_buy = pd.DataFrame(data=self.data.exit_buy.dropna().values, index=self.data.exit_buy.dropna().index,
                                      columns=['price'])

        tradesexit_buy['type'] = 'EXIT_BUY'

        trades = tradesbuy.append([tradessell, tradesexit_sell, tradesexit_buy], ignore_index=False, sort=True)

        self.trades_df = trades.sort_index()
        self.stats['Num of closed trades'] = int(len(self.trades_df) / 2)

    def _calculate_profit_and_loss(self):
        returns = pd.DataFrame(data=self.data.close.diff().fillna(0).values, index=self.data.index, columns=['pnl'])
        pnl = pd.DataFrame(columns=['pnl'])
        if len(self.trades_df) % 2 == 0:
            end_check = len(self.trades_df)
        else:
            end_check = len(self.trades_df) - 1

        for i in range(0, end_check, 2):
            if self.trades_df['type'].iloc[i] == 'BUY':
                pnl_temp = returns.loc[self.trades_df.index[i]:self.trades_df.index[i + 1]].copy()
                pnl_temp.pnl.iloc[0] = pnl_temp.pnl.iloc[0] - self.cost_per_trade
                pnl_temp.pnl.iloc[-1] = pnl_temp.pnl.iloc[-1] - self.cost_per_trade
                if pnl_temp.pnl.cumsum().iloc[-1] >= 0:
                    self.stats['Num of Win trades'] += 1
                else:
                    self.stats['Num of Loss trades'] += 1
                pnl = pnl.append(pnl_temp)

            elif self.trades_df['type'].iloc[i] == 'SELL':
                pnl_temp = -returns.loc[self.trades_df.index[i]:self.trades_df.index[i + 1]].copy()
                pnl_temp.pnl.iloc[0] = pnl_temp.pnl.iloc[0] - self.cost_per_trade
                pnl_temp.pnl.iloc[-1] = pnl_temp.pnl.iloc[-1] - self.cost_per_trade
                if pnl_temp.pnl.cumsum().iloc[-1] >= 0:
                    self.stats['Num of Win trades'] += 1
                else:
                    self.stats['Num of Loss trades'] += 1
                pnl = pnl.append(pnl_temp)
        pnl.pnl = pnl.pnl.cumsum() * self.multiplier
        self.data = self.data.join(pnl, sort=True)
        self.data.pnl = self.data.pnl.ffill()
        self.data.pnl = self.data.pnl.fillna(0)

    def _calculate_maxDrawDown(self):
        i = (np.maximum.accumulate(self.data.pnl) - self.data.pnl).values.argmax()  # end of the period, bottom value
        j = self.data.pnl[:i].values.argmax()  # start of the draw_down , peak value
        self.stats['Max drowdown [%]'] = np.round((self.data.pnl[i] - self.data.pnl[j]) / self.data.pnl[j], 2)
        self.data = self.data.join(
            pd.DataFrame(self.data.pnl[j:i].values, index=self.data.index[j:i], columns=['MAX_DD']), how='outer')

    def _calculate_stats(self):
        self.stats['Win/Loss ratio'] = np.round(self.stats['Num of Win trades'] / self.stats['Num of Loss trades'], 2)
        self.stats['Win rate [%]'] = np.round(self.stats['Num of Win trades'] / self.stats['Num of closed trades'], 2)
        self.stats['Sharp ratio'] = np.round(self.data.pnl.diff().mean() / self.data.pnl.diff().std(), 2)

    def run(self):
        """
        Start Backtest
        """
        self.preprocess()
        for index, row in self.data.iterrows():
            self.strategy(index, row)
        self._create_trades_list()
        self._calculate_profit_and_loss()
        self._calculate_maxDrawDown()
        self._calculate_stats()

    def plot(self, title='Backtest', sub_ind=[], height=700, render=None):
        """
        Plot the results of backtest
        Parameters:
        -----------
        title: title of chart, default = 'Backtest'
        -----------
        sub_ind: format is sub_ind = [[],[]] Each list in sub_ind's list is a subplot. 
        For example you use RSI and Z-score in your strategy and want them to be plotted in subplots under main ohlc plot
        then sub_ind = [['RSI'],['Z-score']]. 'RSI' and 'Z-score' are names of columns in self.data
        -----------
        height: height of chart, default = 700
        -----------
        render: 'browser' for plotting in new browser's tab, default = None
        """
        self.backtest_plot(
            df=self.data.drop(['buy_signal', 'sell_signal', 'exit_buy_signal', 'exit_sell_signal'], axis=1),
            title=title, sub_ind=sub_ind, height=height, render=render)

    def backtest_plot(self, title, df, pnl=True, bars=True, sub_ind=[], height=700, render=None):
        sub_ind = [['pnl', 'MAX_DD']] + sub_ind
        # sub_names = ['Main Chart']
        # for item in sub_ind:
        #    s = ' '
        #    sub_names.append(s.join(item))
        df1 = df.copy()
        fig = make_subplots(rows=len(sub_ind) + 1, cols=1,
                            row_heights=[0.5] + [0.5 / len(sub_ind) for item in sub_ind],
                            shared_xaxes=True,
                            vertical_spacing=0.03)
        # subplot_titles=tuple(sub_names))
        fig.layout.update(height=height, title=title)
        for i in range(len(sub_ind)):
            fig.update_xaxes(type="category", rangeslider=dict(visible=False), row=i + 2, col=1)
            fig.update_xaxes(type="category", rangeslider=dict(visible=False), row=i + 2, col=1)
            for sub in sub_ind[i]:
                fig.add_trace(go.Scatter(x=df1.index, y=df1[sub].values, name=sub), row=i + 2, col=1)
                del df1[sub]

        if bars:
            fig.add_trace(
                go.Candlestick(x=df1.index, open=df1['open'], high=df1.high, low=df1.low, close=df1.close, name='bars'),
                row=1, col=1)
        else:
            fig.add_trace(go.Scatter(x=df1.index, y=df1.close.values, name='close'), row=1, col=1)

        fig.add_trace(go.Scatter(x=df1.index, y=df.buy.values, mode='markers',
                                 marker=dict(color='rgb(12, 255, 12)', size=8,
                                             line=dict(color='rgb(0, 0, 0)', width=1)), name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df1.index, y=df.sell.values, mode='markers',
                                 marker=dict(color='rgb(252, 12, 12)', size=8,
                                             line=dict(color='rgb(0, 0, 0)', width=1)), name='Sell'), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df1.index, y=df.exit_sell.values, mode='markers', marker_color='rgb(0, 0, 0)', marker_size=8,
                       name='exit_sell'), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df1.index, y=df.exit_buy.values, mode='markers', marker_color='rgb(0, 0, 0)', marker_size=8,
                       name='exit_buy'), row=1, col=1)
        del df1['open'], df1['high'], df1['low'], df1['close'], df1['buy'], df1['sell'], df1['exit_buy'], df1[
            'exit_sell']

        for l in range(len(df1.columns)):
            fig.add_trace(go.Scatter(x=df1.index, y=df1.iloc[:, l], name=df1.columns[l]), row=1, col=1)

        fig.update_xaxes(type="category", rangeslider=dict(visible=False), row=1, col=1)
        fig.update_xaxes(type="category", rangeslider=dict(visible=False), row=1, col=1)
        if render:
            fig.show(renderer=render)
        else:
            fig.show()
