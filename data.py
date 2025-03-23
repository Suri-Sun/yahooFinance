import yfinance as yf
from typing import Dict, List, Optional
from datetime import datetime

def get_option_chain(ticker: str, expiration_date: Optional[str] = None) -> Dict:
    """
    Fetch option chain data for a given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        expiration_date (str, optional): Specific expiration date in 'YYYY-MM-DD' format.
                                       If None, returns options for all available dates.
    
    Returns:
        Dict: Dictionary containing calls and puts data for the specified ticker
              Format: {
                  'calls': DataFrame with call options data,
                  'puts': DataFrame with put options data
              }
    
    Raises:
        ValueError: If ticker is invalid or no options data is available
    """
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get options data
        if expiration_date:
            # Verify if the expiration date is available
            if expiration_date not in stock.options:
                available_dates = ', '.join(str(date) for date in stock.options)
                raise ValueError(
                    f"No options available for {expiration_date}. "
                    f"Available dates: {available_dates}"
                )
            
            # Get option chain for specific date
            options = stock.option_chain(expiration_date)
        else:
            # Get the first available expiration date if none specified
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

# Get options for the nearest expiration date
options = get_option_chain('AAPL')

# Get options for a specific expiration date
options = get_option_chain('AAPL', '2025-02-21')

# Access calls and puts data
calls = options['calls']
puts = options['puts']

print(calls)
print(puts)