# Section 1: Import Relevant Libraries. If you Don't Have These Libraries Installed, Follow the Instructions on the Line Below.
# Please click the Windows key, type in "cmd", open "Command Prompt", type in the following code and press enter: pip install yfinance pandas numpy matplotlib
import yfinance as yf # Import yfinance (Yahoo Finance) for downloading historical stock data
import pandas as pd # Import pandas for data manipulation
import numpy as np # Import numpy for numerical operations (e.g. Monte Carlo simulations in the code below)
import matplotlib.pyplot as plt # Import matplotlib for plotting charts
import warnings # Import warnings to suppress unnecessary warnings
# Section 2: Surpresses Future Warnings from yfinance Library.
warnings.simplefilter(action='ignore', category=FutureWarning)
# Section 3: Define your Index Constituents.
# Please change the tickers in the [] to the tickers of the stocks in your index. Ensure they are the Yahoo Finance tickers or the code will crash!
STOCKS = ['PANW', 'FTNT', 'CSCO', 'CHKP', 'CRWD', 'ZS', 'NET', 'CYBR', 'AKAM', 'BAH', 'GEN', 'OKTA', 'QLYS', 'TENB', 'S']
START_DATE = '2018-01-01'
END_DATE = '2025-01-14'
# Section 4: Define the Weightings of the Stocks in your Index.
# Please enter the weightings of the stocks of your index in decimal form, in the same order as the stocks are entered in the STOCKS = [] list above.
CUSTOM_WEIGHTS = [0.15, 0.15, 0.10, 0.07, 0.08, 0.05, 0.05, 0.04, 0.06, 0.07, 0.05, 0.03, 0.04, 0.03, 0.03]
# Safety checks to ensure the number of weights matches number of stocks
if len(CUSTOM_WEIGHTS) != len(STOCKS):
   raise ValueError("Number of weights must match number of tickers")
if not np.isclose(sum(CUSTOM_WEIGHTS), 1): # Ensure weights sum to 1
   raise ValueError("Weights must sum to 1")
# Section 5: Download Historical Adjusted Share Price Data.
data = yf.download(STOCKS, start=START_DATE, end=END_DATE, auto_adjust=True)['Close']
# Section 6: Calculate Daily Returns.
returns = data.pct_change().dropna()
# Section 7: Weighted Index Calculation.
weights = CUSTOM_WEIGHTS
index_returns = returns.dot(weights)
# Section 8: Compute Historical Cumulative Index Value, Starting from a 100 base.
index_value = (1 + index_returns).cumprod() * 100
# Section 9: Run 50 Monte Carlo Simulation Paths.
# Section 9: Run 50 Monte Carlo Simulation Paths.
N_SIMULATIONS = 50
simulated_paths = pd.DataFrame(index=index_value.index)
mu = index_returns.mean()
sigma = index_returns.std()

for i in range(N_SIMULATIONS):
    simulated_daily_returns = np.random.normal(mu, sigma, len(index_returns))
    simulated_index = (1 + simulated_daily_returns).cumprod() * 100
    simulated_paths[f'Sim_{i+1}'] = simulated_index

# Section 10: Performance Summary to be Outputted.
annualized_return = (index_value.iloc[-1]/index_value.iloc[0])**(252/len(index_returns)) - 1
annualized_volatility = index_returns.std() * (252**0.5)
total_return = index_value.iloc[-1]/index_value.iloc[0] - 1
print(f"\nAnnualized Return: {annualized_return*100:.2f}%")
print(f"Annualized Volatility: {annualized_volatility*100:.2f}%")
print(f"Total Return: {total_return*100:.2f}%")
print("Performance summary printed. Chart will pop up next.")
# Section 11: Graphical Output of Backtesting.
plt.figure(figsize=(12, 6))
plt.plot(index_value, label='Historical Index', color='#bc151b', linewidth=2)
plt.plot(simulated_paths, color='#bc151b', alpha=0.3)
plt.title(f' Equity Index Backtest with Monte Carlo Simulations ({N_SIMULATIONS} Runs)')
plt.xlabel('Date')
plt.ylabel('Index Value (Base = 100)')
plt.legend(['Historical Index', 'Simulated Paths'])
plt.grid(True)
plt.show(block=True)