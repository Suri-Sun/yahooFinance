import yfinance as yf
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

def get_option_chain(ticker: str, expiration_date: Optional[str] = None) -> Dict:
    """
    Fetch option chain data for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        expiration_date (str, optional): Specific expiration date in 'YYYY-MM-DD' format.
                                       If None, returns options for the nearest expiration date.
    
    Returns:
        Dict: Dictionary containing calls and puts data for the specified ticker
    """
    try:
        stock = yf.Ticker(ticker)
        
        if expiration_date:
            if expiration_date not in stock.options:
                available_dates = ', '.join(str(date) for date in stock.options)
                raise ValueError(
                    f"No options available for {expiration_date}. "
                    f"Available dates: {available_dates}"
                )
            options = stock.option_chain(expiration_date)
        else:
            if not stock.options:
                raise ValueError(f"No options available for {ticker}")
            options = stock.option_chain(stock.options[0])
        
        return {
            'calls': options.calls,
            'puts': options.puts
        }
    
    except ValueError as e:
        raise e
    except Exception as e:
        raise ValueError(f"Error fetching options data for {ticker}: {str(e)}")

def calculate_fair_variance(ticker: str) -> Dict[str, float]:
    """
    Calculate fair variance for all available option expiration dates.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        Dict[str, float]: Dictionary mapping expiration dates to annualized variance
    """
    stock = yf.Ticker(ticker)
    current_price = stock.history(period='1d')['Close'].iloc[-1]
    risk_free_info = yf.Ticker('^TNX').info
    risk_free = risk_free_info.get('regularMarketPrice', 0) / 100  # Default to 0 if key is missing
    
    variance_term_structure = {}
    
    for expiration in stock.options:
        options = get_option_chain(ticker, expiration)
        calls = options['calls']
        puts = options['puts']
        
        # Calculate time to expiration in years
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        today = datetime.now()
        T = (exp_date - today).days / 365.0
        if T <= 0:
            print(f"Skipping expired option: {expiration}")
            continue
        
        # Sort options by strike
        calls = calls.sort_values('strike').reset_index(drop=True)
        puts = puts.sort_values('strike').reset_index(drop=True)
        
        # Find the strike price closest to current price
        atm_index = (calls['strike'] - current_price).abs().argsort()[0]
        atm_strike = calls.iloc[atm_index]['strike']
        
        # Select OTM options
        otm_calls = calls[calls['strike'] > atm_strike].reset_index(drop=True)
        otm_puts = puts[puts['strike'] < atm_strike].reset_index(drop=True)
        
        # Initialize variance
        variance = 0
        
        # Process OTM calls
        for i in range(len(otm_calls)):
            row = otm_calls.iloc[i]
            K = row['strike']
            
            if i == 0:
                delta_K = K - atm_strike
            else:
                delta_K = K - otm_calls.iloc[i - 1]['strike']
            
            Q = (row['bid'] + row['ask']) / 2  # Mid price
            if np.isnan(Q) or Q <= 0:
                continue  # Skip if bid/ask data is invalid
            
            contribution = (delta_K / K**2) * np.exp(risk_free * T) * Q
            variance += contribution
            
        # Process OTM puts
        for i in range(len(otm_puts)):
            row = otm_puts.iloc[i]
            K = row['strike']
            
            if i < len(otm_puts) - 1:
                delta_K = otm_puts.iloc[i + 1]['strike'] - K
            else:
                delta_K = atm_strike - K
            
            Q = (row['bid'] + row['ask']) / 2  # Mid price
            if np.isnan(Q) or Q <= 0:
                continue  # Skip if bid/ask data is invalid
            
            contribution = (delta_K / K**2) * np.exp(risk_free * T) * Q
            variance += contribution
        
        # Calculate final variance
        try:
            variance = (2 / T) * variance - (1 / T) * ((current_price / atm_strike - 1) ** 2)
            variance_term_structure[expiration] = variance
        except ZeroDivisionError:
            print(f"Division by zero encountered for expiration: {expiration}")
            continue
    
    return variance_term_structure

def plot_variance_term_structure(ticker: str):
    """
    Plot the variance term structure.
    """
    variance_data = calculate_fair_variance(ticker)
    
    if not variance_data:
        print("No variance data to plot.")
        return
    
    # Convert dates to numerical values for plotting
    today = datetime.now()
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in variance_data.keys()]
    times = [(d - today).days / 365.0 for d in dates]
    variances = list(variance_data.values())
    
    # Create smooth interpolation
    if len(times) > 1:
        # Remove duplicate times if any
        times, variances = zip(*sorted(zip(times, variances)))
        
        # Handle cases where all times are the same
        if len(set(times)) > 1:
            f = interpolate.interp1d(times, variances, kind='cubic', fill_value='extrapolate')
            smooth_times = np.linspace(min(times), max(times), 100)
            smooth_variances = f(smooth_times)
            
            plt.plot(times, variances, 'o', label='Observed')
            plt.plot(smooth_times, smooth_variances, '-', label='Interpolated')
        else:
            plt.plot(times, variances, 'o', label='Observed')
    else:
        plt.plot(times, variances, 'o', label='Observed')
    
    plt.xlabel('Time to Expiration (years)')
    plt.ylabel('Fair Variance')
    plt.title(f'Variance Term Structure - {ticker}')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    
    try:
        # Get the variance term structure
        variance_data = calculate_fair_variance(ticker)
        
        if variance_data:
            # Print the results
            print(f"\nVariance Term Structure for {ticker}:")
            print("-" * 50)
            for date, variance in variance_data.items():
                print(f"Expiration: {date}, Variance: {variance:.6f}")
            
            # Plot the term structure
            plot_variance_term_structure(ticker)
        else:
            print("No variance data available.")
    except Exception as e:
        print(f"An error occurred: {e}") 