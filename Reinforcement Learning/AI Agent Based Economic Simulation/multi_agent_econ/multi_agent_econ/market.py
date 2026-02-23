# market.py  (patched, defensive)
from typing import List, Dict, Any, Tuple
import numpy as np

def clear_market(
    orders: List[Dict[str, Any]],
    last_price: float,
    method: str = "batch",
    price_grid: List[float] = None,
    rng = None,
    demand_mult: float = 1.0
) -> Tuple[List[Dict], float, Dict]:
    """
    Defensive batch double-auction clearing.

    Safety measures:
      - convert matched_qty to int and cap by total available volumes
      - always treat per-order qty as int
      - break cleanly if no feasible matches
      - return simple stats for logging
    """
    if price_grid is None:
        price_grid = list(np.linspace(1, 10, 10))

    # Make shallow copies and ensure integer quantities
    buys = []
    sells = []
    for o in orders:
        item = o.copy()
        # force integer qty and price float
        item["qty"] = int(round(float(item.get("qty", 0))))
        item["price"] = float(item.get("price", last_price))
        if item["qty"] <= 0:
            continue
        if item["side"] == "buy":
            buys.append(item)
        else:
            sells.append(item)

    # If no trades possible, return
    if not buys or not sells:
        return [], last_price, {
            "matched_qty": 0,
            "unfilled_buy": sum(b["qty"] for b in buys),
            "unfilled_sell": sum(s["qty"] for s in sells)
        }

    # Precompute total available
    total_buy_qty = sum(b["qty"] for b in buys)
    total_sell_qty = sum(s["qty"] for s in sells)

    # Build aggregated demand & supply by price grid
    demand_by_p = []
    supply_by_p = []
    for p in price_grid:
        demand_qty = sum([b["qty"] for b in buys if b["price"] >= p]) * float(demand_mult)
        supply_qty = sum([s["qty"] for s in sells if s["price"] <= p])
        demand_by_p.append(demand_qty)
        supply_by_p.append(supply_qty)

    matched_volumes = [min(d, s) for d, s in zip(demand_by_p, supply_by_p)]
    best_idx = int(np.argmax(matched_volumes))
    clearing_price = float(price_grid[best_idx])
    matched_qty = matched_volumes[best_idx]

    # Defensive caps: matched_qty cannot exceed actual available integer volume
    # matched_qty might be float (due to demand_mult); cap it and convert to int
    max_possible = min(total_buy_qty, total_sell_qty)
    if np.isnan(matched_qty) or matched_qty <= 0 or max_possible <= 0:
        return [], last_price, {
            "matched_qty": 0,
            "unfilled_buy": int(total_buy_qty),
            "unfilled_sell": int(total_sell_qty)
        }

    # Cap and cast
    matched_qty_int = int(min(max_possible, int(round(float(matched_qty)))))

    # If matched_qty_int is zero after rounding, bail out
    if matched_qty_int <= 0:
        return [], last_price, {
            "matched_qty": 0,
            "unfilled_buy": int(total_buy_qty),
            "unfilled_sell": int(total_sell_qty)
        }

    # Sort orders: buys desc by price, sells asc by price (limit price priority)
    buys_sorted = sorted(buys, key=lambda b: -b["price"])
    sells_sorted = sorted(sells, key=lambda s: s["price"])

    trades: List[Dict[str, Any]] = []
    bi = 0
    si = 0
    remaining = matched_qty_int
    # Greedy matching at clearing_price with safety guards
    while remaining > 0 and bi < len(buys_sorted) and si < len(sells_sorted):
        b = buys_sorted[bi]
        s = sells_sorted[si]

        # If either order is incompatible with clearing_price, move pointers
        if b["price"] < clearing_price:
            bi += 1
            continue
        if s["price"] > clearing_price:
            si += 1
            continue

        # Match feasible quantity
        avail_b = int(b["qty"])
        avail_s = int(s["qty"])
        qty = min(avail_b, avail_s, remaining)
        if qty <= 0:
            # Nothing feasible, advance pointers conservatively
            if avail_b <= 0:
                bi += 1
            if avail_s <= 0:
                si += 1
            # As a last resort, break to avoid infinite loop
            break

        trades.append({
            "buyer": b["agent"],
            "seller": s["agent"],
            "qty": int(qty),
            "price": clearing_price
        })

        # decrement quantities
        b["qty"] = int(b["qty"] - qty)
        s["qty"] = int(s["qty"] - qty)
        remaining -= int(qty)

        if b["qty"] <= 0:
            bi += 1
        if s["qty"] <= 0:
            si += 1

    matched_executed = matched_qty_int - remaining
    # compute unfilleds based on remaining queue
    unfilled_buy = int(sum(b["qty"] for b in buys_sorted[bi:]) + max(0, remaining))
    unfilled_sell = int(sum(s["qty"] for s in sells_sorted[si:]))
    stats = {
        "matched_qty": int(matched_executed),
        "unfilled_buy": unfilled_buy,
        "unfilled_sell": unfilled_sell
    }
    return trades, clearing_price, stats
