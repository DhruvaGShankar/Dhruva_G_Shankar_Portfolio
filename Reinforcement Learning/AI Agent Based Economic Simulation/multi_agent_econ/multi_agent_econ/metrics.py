# metrics_extended.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

def gini_coeff(x: List[float]) -> float:
    arr = np.array(x, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    if np.any(arr < 0):
        arr = arr - arr.min() + 1e-12
    arr = np.sort(arr)
    n = arr.size
    cum = arr.cumsum()
    if cum[-1] == 0:
        return 0.0
    gini = (2.0 * (np.arange(1, n+1) * arr).sum()) / (n * cum[-1]) - (n + 1) / n
    return float(gini)

def rolling_volatility(price_series: List[float], window: int = 10) -> np.ndarray:
    p = np.array(price_series, dtype=float)
    if p.size < window:
        return np.zeros_like(p)
    vol = np.full_like(p, fill_value=np.nan, dtype=float)
    for i in range(window-1, len(p)):
        vol[i] = np.std(p[i-window+1:i+1])
    # fill initial nan with first computable vol
    nan_mask = np.isnan(vol)
    if np.any(~nan_mask):
        first = np.where(~nan_mask)[0][0]
        vol[:first] = vol[first]
    else:
        vol[:] = 0.0
    return vol

def compute_shortage_ratio_by_step(infos_series: List[Dict[str, Any]]) -> np.ndarray:
    """
    infos_series: list of step-level infos (one element per step). Each element must contain clearing_stats
    returns shortage_ratio_t = unfilled_buy / (unfilled_buy + matched_qty)  (0 if denom 0)
    NOTE: your env returns per-agent infos; assume we pass the infos of one agent per step (they contain clearing_stats)
    """
    ratios = []
    for info in infos_series:
        stats = info.get("clearing_stats", {})
        unfilled_buy = stats.get("unfilled_buy", 0)
        matched = stats.get("matched_qty", 0)
        denom = unfilled_buy + matched
        if denom <= 0:
            ratios.append(0.0)
        else:
            ratios.append(float(unfilled_buy) / float(denom))
    return np.array(ratios, dtype=float)

# ---- multi-run aggregation helpers ----

def aggregate_runs(runs: List[List[float]]) -> Dict[str, np.ndarray]:
    """
    runs: list of series (each series has same length)
    returns dict with mean, lower (2.5%), upper (97.5%), std
    """
    arr = np.array(runs, dtype=float)  # shape (n_runs, T)
    mean = np.nanmean(arr, axis=0)
    lower = np.nanpercentile(arr, 2.5, axis=0)
    upper = np.nanpercentile(arr, 97.5, axis=0)
    std = np.nanstd(arr, axis=0)
    return {"mean": mean, "lower": lower, "upper": upper, "std": std, "raw": arr}

# ---- plotting ----

def plot_with_ci(series_dict: Dict[str, Dict[str, np.ndarray]], ylabel: str, title: str, fname: str = None):
    """
    series_dict: {"label1": {"mean":..., "lower":..., "upper":...}, "label2": {...}}
    """
    plt.figure(figsize=(8,4))
    for label, stats in series_dict.items():
        x = np.arange(len(stats["mean"]))
        plt.plot(x, stats["mean"], label=label, linewidth=1.6)
        plt.fill_between(x, stats["lower"], stats["upper"], alpha=0.2)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=200)
    else:
        plt.show()
    plt.close()

def comparison_panel_plot(results: Dict[str, Dict[str, Any]], outdir: str):
    """
    results: mapping policy_name -> { "prices": list of runs (each run list), "trade_values": [...], "shortage": [...], "gini": [...] }
    Saves 4 subplot figure with price, gini, GDP-proxy, shortage ratio.
    """
    import os
    os.makedirs(outdir, exist_ok=True)

    # aggregate each series
    agg = {}
    for policy, data in results.items():
        agg[policy] = {
            "prices": aggregate_runs(data["prices"]),
            "gini": aggregate_runs(data["gini"]),
            "gdp": aggregate_runs(data["gdp"]),
            "shortage": aggregate_runs(data["shortage"]),
            "trade_vol": aggregate_runs(data["trade_vol"])
        }

    T = agg[next(iter(agg))]["prices"]["mean"].shape[0]
    x = np.arange(T)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    # Price
    ax = axs[0,0]
    for policy, stats in agg.items():
        ax.plot(x, stats["prices"]["mean"], label=policy)
        ax.fill_between(x, stats["prices"]["lower"], stats["prices"]["upper"], alpha=0.15)
    ax.set_title("Price Evolution")
    ax.set_xlabel("Step"); ax.set_ylabel("Price"); ax.legend()

    # Gini
    ax = axs[0,1]
    for policy, stats in agg.items():
        ax.plot(x, stats["gini"]["mean"], label=policy)
        ax.fill_between(x, stats["gini"]["lower"], stats["gini"]["upper"], alpha=0.15)
    ax.set_title("Gini over Time")
    ax.set_xlabel("Step"); ax.set_ylabel("Gini")

    # GDP proxy (cumulative)
    ax = axs[1,0]
    for policy, stats in agg.items():
        ax.plot(x, np.cumsum(stats["gdp"]["mean"]), label=policy)
        ax.fill_between(x, np.cumsum(stats["gdp"]["lower"]), np.cumsum(stats["gdp"]["upper"]), alpha=0.15)
    ax.set_title("Cumulative GDP Proxy (trade value)")
    ax.set_xlabel("Step"); ax.set_ylabel("Cumulative Value")

    # Shortage ratio
    ax = axs[1,1]
    for policy, stats in agg.items():
        ax.plot(x, stats["shortage"]["mean"], label=policy)
        ax.fill_between(x, stats["shortage"]["lower"], stats["shortage"]["upper"], alpha=0.15)
    ax.set_title("Shortage Ratio over Time")
    ax.set_xlabel("Step"); ax.set_ylabel("Unfilled fraction")

    plt.tight_layout()
    fpath = f"{outdir}/comparison_panel.png"
    plt.savefig(fpath, dpi=200)
    plt.close()
    return fpath
