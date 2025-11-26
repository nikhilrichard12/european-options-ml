import numpy as np
from scipy.stats import norm
import pandas as pd
from google.colab import files

def Black_Scholes_call(S0, K, T, r, sigma, q):
    """
    Calculate Black-Scholes call option price
    
    Parameters:
    S0: Current stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Implied volatility
    q: Dividend yield
    """
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    call_price = S0*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    return call_price

def Black_Scholes_put(S0, K, T, r, sigma, q):
    """
    Calculate Black-Scholes put option price
    
    Parameters:
    S0: Current stock price
    K: Strike price
    T: Time to maturity
    r: Risk-free rate
    sigma: Implied volatility
    q: Dividend yield
    """
    d1 = (np.log(S0/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = (np.log(S0/K) + (r - q - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S0*np.exp(-q*T)*norm.cdf(-d1)
    
    return put_price

def BS_call_synthetic_data(S0, N):
    """
    Generate synthetic call option data using Black-Scholes
    
    Parameters:
    S0: Initial stock price
    N: Number of samples to generate
    """
    S = np.full(N, S0)
    K = np.random.uniform(90, 110, N)
    q = np.random.uniform(0.02, 0.06, N)
    r = np.random.uniform(0.01, 0.06, N)
    sigma = np.random.uniform(0.15, 0.3, N)
    T = np.random.uniform(0.25, 2, N)
    
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    df = pd.DataFrame({
        'Stock Price (S)': S,
        'Strike Price (K)': K,
        'Dividend Yield (q)': q,
        'Risk-Free Rate (r)': r,
        'Implied Volatility (v)': sigma,
        'Time to Expiry (T)': T,
        'Call Option Price': call_price
    })
    
    return df

# Example usage
if __name__ == "__main__":
    # Single option pricing example
    S = 100  # stock price
    K = 105  # strike price
    T = 0.5  # time to maturity
    r = 0.05  # risk-free rate
    sigma = 0.1  # implied volatility
    q = 0.02  # continuous dividend yield
    
    call_price = Black_Scholes_call(S, K, T, r, sigma, q)
    put_price = Black_Scholes_put(S, K, T, r, sigma, q)
    
    print("Call price: ", call_price)
    print("Put price: ", put_price)
    
    # Generate synthetic data
    S0 = 100
    N = 1000
    
    df = BS_call_synthetic_data(S0, N)
    print("\nSynthetic data sample:")
    print(df.head())
    
    # Note: Google Colab file download functionality removed for local use
    # If needed, save with: df.to_csv("BSM_call_sample.csv", index=False)