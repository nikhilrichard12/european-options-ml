# SPY Option Pricing with ML

Pricing SPY call options using ML vs. Black-Scholes(1973 mathematical framework used to calculate the theoretical fair value of European-style options). Based on Q1 2023 data.

## Results

ML models outperformed the baseline. Random Forest was the most accurate.

| Model | R2 | MAE |
| :--- | :--- | :--- |
| **Random Forest** | 99.63% | $1.60 |
| **XGBoost** | 99.54% | $1.85 |
| **Neural Network** | 99.12% | $2.60 |
| **Black-Scholes** | 97.12% | $3.18 |

## Model Logic

* **Random Forest:** 100 trees, depth 20. Strike price = 93% importance.
* **XGBoost:** 100 trees, 0.1 learning rate. Gradient boosting for sequential error correction.
* **Neural Network:** 3-layer (64-32-1) with ReLU and Adam.
* **Black-Scholes:** Benchmark. Fails because it assumes constant volatility.


## Data and Features

* **Source:** OptionDX EOD SPY calls (Q1 2023).
* **Samples:** ~190k contracts; 80/20 train-test split.
* **Inputs:** Underlying price, strike, DTE, IV, and Greeks (Delta, Gamma, Vega, Theta).

---

**Nikhil Richard** CS + Econ @ UIUC
