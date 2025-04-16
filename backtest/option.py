import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

class BacktestOptions:
    def __init__(self, ticker, strategy, short_put_percentage, long_put_percentage, short_put_premium, long_put_premium, rolling_weeks=2):
        self.ticker = ticker
        self.strategy = strategy
        self.short_put_percentage = short_put_percentage
        self.long_put_percentage = long_put_percentage
        self.short_put_premium = short_put_premium
        self.long_put_premium = long_put_premium
        self.rolling_weeks = rolling_weeks
        self.data = None
        self.performance = None

    def fetch_data(self):
        """Fetch historical stock data from Yahoo Finance."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Last 6 months
        self.data = yf.download(self.ticker, start=start_date, end=end_date)
        if self.data.empty:
            raise ValueError(f"No data found for {self.ticker}.")

    def calculate_performance(self):
        """Calculate the performance based on the selected strategy."""
        self.data['Date'] = self.data.index
        self.data.set_index('Date', inplace=True)
        fridays = self.data[self.data.index.weekday == 4]  # Get only Fridays

        self.performance = pd.Series(dtype=float)

        for i in range(len(fridays) - 1):
            current_friday = fridays.index[i]
            next_friday = fridays.index[i + 1]

            current_price = self.data.loc[current_friday]['Close']
            short_put_strike = current_price * (1 - self.short_put_percentage)
            long_put_strike = current_price * (1 - self.long_put_percentage)

            period_data = self.data.loc[current_friday:next_friday]

            if self.strategy == 'spread':
                self.performance = self.performance.add(self.calculate_put_spread_performance(period_data, short_put_strike, long_put_strike), fill_value=0)
            elif self.strategy == 'straddle':
                self.performance = self.performance.add(self.calculate_straddle_performance(period_data, short_put_strike, long_put_strike), fill_value=0)

    def calculate_put_spread_performance(self, data, short_put_strike, long_put_strike):
        """Calculate the performance of a put spread strategy."""
        data['Short Put'] = np.where(data['Close'] < short_put_strike, short_put_strike - data['Close'], 0) - self.short_put_premium
        data['Long Put'] = np.where(data['Close'] < long_put_strike, long_put_strike - data['Close'], 0) - self.long_put_premium
        data['Total'] = data['Short Put'] - data['Long Put']
        return data['Total'].cumsum()

    def calculate_straddle_performance(self, data, put_strike, call_strike):
        """Calculate the performance of a straddle strategy."""
        data['Put'] = np.where(data['Close'] < put_strike, put_strike - data['Close'], 0) - self.short_put_premium
        data['Call'] = np.where(data['Close'] > call_strike, data['Close'] - call_strike, 0) - self.long_put_premium
        data['Total'] = data['Put'] + data['Call']
        return data['Total'].cumsum()

    def calculate_stats(self):
        """Calculate essential statistics."""
        returns = self.performance.pct_change().dropna()
        annualized_return = np.mean(returns) * 252  # Assuming 252 trading days
        annualized_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility

        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()

        return {
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        }

    def plot_results(self):
        """Plot the results of the backtest."""
        if self.data is None or self.performance is None:
            print("No data to plot.")
            return

        plt.figure(figsize=(14, 7))
        plt.plot(self.data.index, self.data['Close'], label=f'{self.ticker} Stock Price', color='blue')
        plt.plot(self.data.index, self.performance, label='Strategy Performance', color='orange')
        plt.title(f'{self.ticker} {self.strategy.capitalize()} Backtest Performance')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid()
        plt.show()

def main():
    # Parameters for the backtest
    ticker = 'TSLA'
    strategy = 'spread'  # Change to 'straddle' for straddle strategy
    short_put_percentage = 0.45  # 45% below the current price
    long_put_percentage = 0.55     # 55% below the current price
    short_put_premium = 50         # Example short put premium
    long_put_premium = 20          # Example long put premium

    # Create a backtest instance
    backtest = BacktestOptions(ticker, strategy, short_put_percentage, long_put_percentage, short_put_premium, long_put_premium)

    # Fetch data and calculate performance
    backtest.fetch_data()
    backtest.calculate_performance()

    # Calculate and print statistics
    stats = backtest.calculate_stats()
    print(f"Performance Statistics for {ticker} ({strategy.capitalize()}):")
    for key, value in stats.items():
        print(f"{key}: {value:.4f}")

    # Plot results
    backtest.plot_results()

if __name__ == "__main__":
    main()
