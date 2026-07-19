# Topological Data Analysis (TDA) Market Regime Discovery

This repository implements a state-of-the-art quantitative trading framework that leverages **Algebraic Topology** (specifically Persistent Homology) and **Machine Learning** to detect structural market regime shifts and impending crashes. By translating financial time series into geometric spaces, the model captures the "shape" of the market to drive dynamic asset allocation.

## Overview

Traditional quantitative models often rely on moving averages or variance-covariance matrices, which suffer from lag and non-stationarity. This project takes a completely different approach:
1. **Geometric Translation:** It computes rolling cross-sectional correlation distance matrices for a basket of assets (e.g., Top-5 Cryptocurrencies).
2. **Vietoris-Rips Complexes:** Uses `giotto-tda` to extract topological features ($H_0$ connected components and $H_1$ persistence loops), representing the fragmentation and "holes" in market correlations.
3. **Machine Learning Regime Classification:** Translates persistence diagrams into mathematically bounded `PersistenceImage` feature vectors and feeds them into a strictly causal Random Forest to classify the market regime (Normal vs. Crash).
4. **Live Execution:** Automatically connects to Binance to poll real-time data, update the topological geometry, and allocate capital defensively when a crash structure forms.

## Repository Structure

* `main.py`: The entry point for historical backtesting across different eras (e.g., Dot-Com, 2008 GFC, 2020 COVID-19) for ETF data.
* `tda_pipeline.py`: The core topological engine. Handles correlation matrix transformation, persistence diagram extraction, noise filtering, and Persistence Image vectorization.
* `ml_regime_model.py`: Implements Unsupervised (K-Means) and Supervised (Random Forest with Purged Cross-Validation) models to translate topology into asset allocation.
* `train_crypto_model.py`: Training script that fetches historical 1-hour cryptocurrency data from Binance, calibrates the topological bounds, and persists the `.joblib` models.
* `live_trader.py`: The out-of-sample live execution engine. Runs an infinite polling loop connected to Binance, evaluating real-time topology to simulate paper trades.
* `data_loader.py`: Utility functions for fetching data via Yahoo Finance.
* `cross_sectional_alpha.py`: Module for extracting cross-sectional alpha (Topological Beta Neutralization) based on $L_1$ norms.

## Installation

This project requires Python 3.10+ and heavy numerical/topological libraries.

```bash
# Clone the repository
git clone https://github.com/CodeByRachit/TDA-Market-Regime-Discovery.git
cd TDA-Market-Regime-Discovery

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn giotto-tda yfinance ccxt joblib matplotlib
```

## Usage

### 1. Training the Cryptocurrency Model
To run the live trader, you must first train the topological Random Forest on historical 1-hour cryptocurrency data (2020-2023).

```bash
python train_crypto_model.py
```
*This fetches ~30,000 candles from Binance, computes rolling distance matrices, extracts topological features, trains the model with balanced class weights, and saves `rf_model.joblib` and `pi_transformer.joblib`.*

### 2. Live Paper Trading
Start the live paper trading engine to monitor the market in real-time.

```bash
python -u live_trader.py
```
*The script initializes a simulated $10,000 portfolio, polls Binance every hour, re-evaluates the topological geometry, and executes trades (deducting a 0.1% taker fee) when regime shifts are detected.*

### 3. Historical Backtesting
To evaluate the strategy against historical structural crashes (e.g., 2020 COVID-19):

```bash
python main.py --mode ml_macro --era 2020
```

## Strategy Results & Visualization

When you run the historical backtesting and high-frequency PoC scripts, the engine will automatically generate visualizations that demonstrate how Algebraic Topology identifies structural failure.

1. **Macro Regime Model (`macro_results_2020_supervised_ml.png`)**
   - **Topological Early Warning Signal (Bottom)**: Shows the $L_1$ norm of the $1$-dimensional persistence landscape spiking dynamically during the COVID-19 crash.
   - **Dynamic Asset Allocation (Middle)**: The Random Forest perfectly translates this topological spike into a defensive maneuver, actively dropping equity exposure from 100% to 20%.
   - **Strategy Wealth (Top)**: Proves that dodging the structural crash geometrically prevents massive drawdowns and significantly outperforms a static benchmark.
<img width="1400" height="1200" alt="image" src="https://github.com/user-attachments/assets/8c1fb271-f914-4092-bfd4-7e6ec26ae009" />

2. **High-Frequency Market Microstructure (`hft_poc_results.png`)**
   - Shows the engine running on 1-minute intraday tick/candle data with an adaptive noise filtering threshold (`epsilon=0.0`). The topological fragmentation index isolates microscopic order book imbalances and transient liquidity vacuums milliseconds before violent price drops.
<img width="1400" height="600" alt="image" src="https://github.com/user-attachments/assets/909eaed8-24c5-43df-b9aa-97319219692e" />

## Key Technical Features

* **Strict Causality Bounding:** The `PersistenceImage` grid is physically locked to the absolute maximums of the correlation distance metric ($0$ to $\approx 1.415$) to prevent lookahead bias from future volatility clusters.
* **Class Imbalance Handling:** The Random Forest utilizes `class_weight="balanced"` to heavily penalize missing rare crash events (which constitute only 2-4% of market history).
* **Phantom Fee Elimination:** The live execution loop maintains state, only deducting transaction fees when the topological regime explicitly shifts the target allocation.

## Disclaimer
*This repository is for research and educational purposes only. The live execution script simulates paper trades and does not place real orders. Cryptocurrencies and financial markets are highly volatile.*
