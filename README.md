# 📈 Markowitz 2D Portfolio Optimizer

A lightweight desktop application built with Python and Tkinter for Modern Portfolio Theory (MPT) mean-variance optimization and efficient frontier exploration.

The tool calculates daily percentage returns over a 1-year historical window, computes expected returns and the population covariance matrix, and solves for optimal asset allocations using Sequential Least Squares Programming (SLSQP).

---

## Mathematical Formulation

The model implements standard matrix algebra for portfolio risk and return:

* **Expected Portfolio Return:**
  $$E[R_p] = \mathbf{w}^T \boldsymbol{\mu}$$

* **Portfolio Variance:**
  $$\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$$

* **Portfolio Volatility (Risk):**
  $$\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}$$

* **Optimization Constraints:**
  $$\sum_{i=1}^n w_i = 1, \quad 0 \le w_i \le 1 \quad \forall i$$

Where:
* $\mathbf{w}$ is the vector of asset portfolio weights.
* $\boldsymbol{\mu}$ is the vector of annualized expected returns ($\sum R_i$).
* $\boldsymbol{\Sigma}$ is the population covariance matrix ($\text{ddof} = 0$).

---

## Features

* **Data Ingestion:** Fetch historical price data directly via Yahoo Finance (`yfinance`) or load local Excel spreadsheets (`.xlsx`).
* **Portfolio Optimization:**
  * **Minimum Volatility:** Solves $\min \sigma_p$ subject to fully invested weights.
  * **Maximum Return:** Solves $\max E[R_p]$ across allowable bounds.
* **2D Efficient Frontier & Curve Interpolation:**
  * Generates 11 discrete weight steps across assets.
  * Plots $\sigma_p$ vs. $E[R_p]$ with B-spline smoothing (`scipy.interpolate.make_interp_spline`) integrated via Matplotlib Canvas.
* **Summary Analytics:** Displays individual asset daily/annual means, standard deviations, variances, and the complete covariance matrix.

---

## Requirements

Ensure Python 3.8+ is installed, then install dependencies:

```bash
pip install numpy pandas scipy matplotlib yfinance openpyxl# markowitz_optimizer
