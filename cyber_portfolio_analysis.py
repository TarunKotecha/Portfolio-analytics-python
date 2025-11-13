#!/usr/bin/env python3
"""
Cybersecurity Portfolio Analysis (robust yfinance handling)

- Uses auto_adjust=True so we can rely on 'Close' (already adjusted).
- Handles yfinance returning single/multi-index columns.
- Drops tickers that fail to download and renormalizes weights.
- Converts .L (GBp) listings to USD using GBPUSD=X.
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception as e:
    raise SystemExit(
        "This script requires the 'yfinance' package.\n"
        "Install with:  py -m pip install yfinance pandas numpy matplotlib\n"
        f"Import error: {e}"
    )

# ------------------------ Portfolio weights ------------------------
DEFAULT_WEIGHTS: Dict[str, float] = {
    "PANW": 0.16,
    "FTNT": 0.14,
    "CSCO": 0.11,
    "CHKP": 0.08,
    "CRWD": 0.07,
    "ZS":   0.04,
    "NET":  0.04,
    "CYBR": 0.05,
    "AKAM": 0.07,
    "BAH":  0.04,
    "GEN":  0.03,
    "OKTA": 0.03,
    "QLYS": 0.03,
    "TENB": 0.02,
    "CASH": 0.08,
}


# ------------------------ Helpers ------------------------
def validate_weights(w: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(w.values()))
    if total <= 0:
        raise ValueError("Weights must sum to a positive number.")
    if not np.isclose(total, 1.0):
        print(f"[WARN] Weights sum to {total:.6f}; normalizing to 1.0")
        w = {k: v / total for k, v in w.items()}
    return w


def _extract_price_table(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Return a (date x tickers) table of prices.
    Works when yfinance groups by column OR by ticker, and when only 'Close' exists.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Case A: level 0 are fields (Close, Open, etc.)
        if "Close" in df.columns.get_level_values(0):
            prices = df["Close"].copy()
        # Case B: level 0 are tickers, level 1 are fields
        elif "Close" in df.columns.get_level_values(1):
            prices = df.xs("Close", axis=1, level=1).copy()
        else:
            raise KeyError("Could not find a 'Close' column in yfinance data.")
    else:
        # Single-ticker download -> plain columns
        if "Close" in df.columns:
            prices = df[["Close"]].copy()
        elif "Adj Close" in df.columns:
            prices = df[["Adj Close"]].copy()
        else:
            prices = df.select_dtypes(include="number").iloc[:, [0]].copy()
        prices.columns = [tickers[0]]
    return prices


def download_adjusted_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    # auto_adjust=True -> use 'Close' already adjusted for splits/dividends
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    prices = _extract_price_table(raw, tickers)
    return prices.resample("ME").last()


def convert_to_usd(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert .L tickers (GBp) to USD using GBPUSD=X."""
    london = [c for c in prices.columns if c.endswith(".L")]
    if not london:
        return prices

    fx_raw = yf.download(
        "GBPUSD=X",
        start=prices.index.min() - pd.Timedelta(days=7),
        end=prices.index.max() + pd.Timedelta(days=7),
        auto_adjust=True,
        progress=False,
    )

    # pick usable series
    if isinstance(fx_raw, pd.DataFrame):
        if "Close" in fx_raw.columns:
            fx_series = fx_raw["Close"]
        elif "Adj Close" in fx_raw.columns:
            fx_series = fx_raw["Adj Close"]
        else:
            fx_series = fx_raw.select_dtypes(include="number").iloc[:, 0]
    else:
        fx_series = fx_raw

    fx_m = fx_series.resample("M").last().reindex(prices.index).ffill()

    out = prices.copy()
    for tic in london:
        out[tic] = (prices[tic] * 0.01) * fx_m  # GBp -> GBP -> USD
    return out


def portfolio_monthly_returns(prices_usd_m: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    prices = prices_usd_m[sorted(weights.keys())].dropna(how="any")
    rets = prices.pct_change().dropna()
    w = pd.Series(weights).reindex(prices.columns).astype(float)
    port_ret = rets.dot(w)
    port_ret.name = "Portfolio"
    return port_ret


def compute_drawdown(cum: pd.Series) -> pd.Series:
    return (cum / cum.cummax()) - 1.0


def gain_to_pain_ratio(returns: pd.Series) -> float:
    pos = returns[returns > 0].sum()
    neg = -returns[returns < 0].sum()
    return float(np.inf if neg == 0 else pos / neg)


def trimmed_mean(returns: pd.Series, trim: float = 0.10) -> float:
    n = len(returns)
    k = int(np.floor(n * trim))
    s = returns.sort_values().iloc[k:n - k] if k > 0 else returns.sort_values()
    return float(s.mean())


@dataclass
class Metrics:
    cagr: float
    ann_return_arith: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    calmar: float
    var_5_m: float
    median_annual_return: float
    trimmed_avg_m: float
    gain_to_pain: float


def compute_metrics(monthly_returns: pd.Series, rf_annual: float = 0.02) -> Metrics:
    cum = (1 + monthly_returns).cumprod()
    months = monthly_returns.shape[0]
    years = months / 12.0 if months > 0 else np.nan
    cagr = cum.iloc[-1] ** (1 / years) - 1 if months > 0 else np.nan
    mean_m = monthly_returns.mean()
    vol_m = monthly_returns.std()
    ann_return_arith = mean_m * 12.0
    ann_vol = vol_m * np.sqrt(12.0)
    sharpe = (ann_return_arith - rf_annual) / ann_vol if ann_vol > 0 else np.nan
    dd = compute_drawdown(cum)
    max_dd = float(dd.min())
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan
    var_5_m = float(np.percentile(monthly_returns, 5))
    yr = monthly_returns.to_frame("m")
    yr["year"] = yr.index.year
    annual_comp = yr.groupby("year")["m"].apply(lambda x: (1 + x).prod() - 1)
    median_annual = float(annual_comp.median()) if len(annual_comp) else np.nan
    trimmed_avg_m = float(trimmed_mean(monthly_returns, 0.10))
    g2p = gain_to_pain_ratio(monthly_returns)
    return Metrics(
        cagr=float(cagr),
        ann_return_arith=float(ann_return_arith),
        ann_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        calmar=float(calmar),
        var_5_m=float(var_5_m),
        median_annual_return=float(median_annual),
        trimmed_avg_m=float(trimmed_avg_m),
        gain_to_pain=float(g2p),
    )


def metrics_to_frame(name: str, m: Metrics) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CAGR": m.cagr,
            "Annualized Return (arith)": m.ann_return_arith,
            "Annualized Volatility": m.ann_vol,
            "Sharpe (annual, rf)": m.sharpe,
            "Max Drawdown": m.max_drawdown,
            "Calmar": m.calmar,
            "5% Monthly VaR": m.var_5_m,
            "Median Annual Return": m.median_annual_return,
            "Trimmed Avg Return (monthly)": m.trimmed_avg_m,
            "Gain-to-Pain Ratio": m.gain_to_pain,
        },
        index=[name],
    ).T


# ------------------------ MAIN ------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze a custom cybersecurity portfolio.")
    parser.add_argument("--start", default="2020-11-03")
    parser.add_argument("--end", default="2025-11-03")
    parser.add_argument("--rf", type=float, default=0.0441, help="Annual risk-free rate (e.g., 0.0441)")
    parser.add_argument("--benchmark", default="SPY", help="Optional benchmark ticker")
    args = parser.parse_args()

    weights = validate_weights(DEFAULT_WEIGHTS.copy())

# --- NEW CODE START ---
# Pull out any cash weight so it’s not treated as a ticker
cash_w = float(weights.pop("CASH", 0.0))
# --- NEW CODE END ---

tickers = list(weights.keys())

print("[INFO] Downloading prices...")
prices_m_local = download_adjusted_prices(tickers, args.start, args.end)

    # Drop failed tickers and renormalize weights
    bad_cols = [c for c in prices_m_local.columns if prices_m_local[c].isna().all()]
    if bad_cols:
        print(f"[WARN] Missing/failed tickers dropped: {bad_cols}")
        for c in bad_cols:
            prices_m_local = prices_m_local.drop(columns=c)
            if c in weights:
                weights.pop(c)
        weights = validate_weights(weights)
        tickers = list(weights.keys())

    prices_m_usd = convert_to_usd(prices_m_local)

    port_m = portfolio_monthly_returns(prices_m_usd, weights)
# --- CASH BLEND START ---
    if cash_w > 0:
    # convert annual rf to monthly
    monthly_rf = (1.0 + args.rf) ** (1.0 / 12.0) - 1.0
    cash_series = pd.Series(monthly_rf, index=port_m.index, name="CASH")
    # combine equities (sum to 1 - cash_w) with cash
    port_m = (1.0 - cash_w) * port_m + cash_w * cash_series
# --- CASH BLEND END ---

    prices_m_usd = prices_m_usd.reindex(port_m.index)
    prices_m_usd.to_csv("prices_usd.csv")
    cons_rets = prices_m_usd.pct_change().dropna()
    cons_rets["Portfolio"] = port_m.reindex(cons_rets.index)
    cons_rets.to_csv("monthly_returns.csv")

    pm = compute_metrics(port_m, rf_annual=args.rf)
    metrics_df = metrics_to_frame("Portfolio", pm)
    metrics_df.to_csv("portfolio_metrics.csv")
    print("\nPortfolio metrics:\n", metrics_df.round(4))

    if args.benchmark:
        print(f"[INFO] Downloading benchmark: {args.benchmark}")
        bench_p = download_adjusted_prices([args.benchmark], args.start, args.end)
        bench_p_usd = convert_to_usd(bench_p)
        bench_r = bench_p_usd.pct_change().dropna().iloc[:, 0]
        bm = compute_metrics(bench_r, rf_annual=args.rf)
        metrics_to_frame(args.benchmark, bm).to_csv("portfolio_vs_benchmark.csv")
        print("\nBenchmark metrics saved to portfolio_vs_benchmark.csv")

    cum_port = (1 + port_m).cumprod()
    dd = compute_drawdown(cum_port)

    plt.figure()
    cum_port.plot(title="Portfolio: Growth of $1")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.savefig("cumulative_value.png", dpi=144)
    plt.close()

    plt.figure()
    dd.plot(title="Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig("drawdown.png", dpi=144)
    plt.close()

    print("\nSaved files:\n - portfolio_metrics.csv\n - portfolio_vs_benchmark.csv (if benchmark provided)\n - monthly_returns.csv\n - prices_usd.csv\n - cumulative_value.png\n - drawdown.png\nDone.")


if __name__ == "__main__":
    main()
