# agents.py
import numpy as np
from typing import Tuple

# Each heuristic returns (price_idx, qty, side) where side: 0=hold,1=buy,2=sell

def consumer_heuristic(obs: np.ndarray, env, epsilon=0.12) -> Tuple[int, int, int]:
    # obs layout: [cash, inv, last_price, ema_d, ema_s, time_frac, risk]
    cash, inv, last_p, ema_d, ema_s, time_frac, risk = obs
    # exploration
    if env.rng.random() < epsilon:
        # random small action
        price_idx = env.rng.integers(0, len(env.price_grid))
        qty = int(env.rng.integers(0, min(env.max_qty, max(1, int(cash // max(1e-6, last_p)))) + 1))
        side = int(env.rng.choice([0, 1, 2]))
        return (price_idx, qty, side)

    # prefer to buy when demand > supply and affordable
    if ema_d > ema_s and cash >= last_p:
        price_idx = min(len(env.price_grid)-1, max(0, int(round(last_p))-1))
        qty = 1
        side = 1
    else:
        price_idx = max(0, min(len(env.price_grid)-1, int(round(last_p))-1))
        qty = 0
        side = 0
    return (price_idx, qty, side)

def producer_heuristic(obs: np.ndarray, env, epsilon=0.08) -> Tuple[int, int, int]:
    cash, inv, last_p, ema_d, ema_s, time_frac, risk = obs
    # exploration
    if env.rng.random() < epsilon:
        price_idx = env.rng.integers(0, len(env.price_grid))
        qty = int(env.rng.integers(0, min(env.max_qty, int(inv)) + 1))
        side = int(env.rng.choice([0, 1, 2]))
        return (price_idx, qty, side)

    prod_cost = env.params.get(env.params.keys().__iter__().__next__(), {}).get("prod_cost", 2.0)
    # If market price > prod_cost, and we have inventory, sell
    if last_p > prod_cost and inv >= 1:
        price_idx = min(len(env.price_grid)-1, max(0, int(round(last_p))-1))
        qty = 1
        side = 2
    else:
        price_idx = max(0, min(len(env.price_grid)-1, int(round(last_p))-1))
        qty = 0
        side = 0
    return (price_idx, qty, side)

def trader_heuristic(obs: np.ndarray, env, epsilon=0.10, momentum_window=3) -> Tuple[int, int, int]:
    cash, inv, last_p, ema_d, ema_s, time_frac, risk = obs
    # exploration
    if env.rng.random() < epsilon:
        price_idx = env.rng.integers(0, len(env.price_grid))
        qty = int(env.rng.integers(0, min(env.max_qty, int(max(1, inv + 1))) ) )
        side = int(env.rng.choice([0, 1, 2]))
        price_idx = max(0, min(len(env.price_grid)-1, price_idx))
        return (price_idx, qty, side)

    prices = env.price_series[-momentum_window:] if len(env.price_series) >= momentum_window else env.price_series
    if len(prices) >= 2 and prices[-1] > prices[-2] and cash >= last_p:
        # buy on momentum
        price_idx = min(len(env.price_grid)-1, int(round(last_p)))
        qty = 1
        side = 1
    elif len(prices) >= 2 and prices[-1] < prices[-2] and inv >= 1:
        price_idx = max(0, int(round(last_p))-1)
        qty = 1
        side = 2
    else:
        price_idx = max(0, int(round(last_p))-1)
        qty = 0
        side = 0
    price_idx = max(0, min(len(env.price_grid)-1, price_idx))
    return (price_idx, qty, side)
