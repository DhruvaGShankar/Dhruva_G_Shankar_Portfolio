# experiments_runner.py
import os
import numpy as np
from multi_agent_econ.econ_env import EconEnv
from multi_agent_econ.agents import consumer_heuristic, producer_heuristic, trader_heuristic
from multi_agent_econ.metrics import gini_coeff, rolling_volatility, compute_shortage_ratio_by_step
import json
from typing import List, Dict, Any

def single_run(agent_configs, policy_params, seed=0, max_steps=300):
    env = EconEnv(agent_configs, max_steps=max_steps, price_grid=list(range(1, 11)),
                  policy_params=policy_params, seed=seed,
                  reward_config={"type": "wealth_plus_consumption", "consumption_weight": 1.0, "inv_penalty": 0.0})
    obs, _ = env.reset()
    # storage per step
    prices = []
    trade_values = []
    trade_vol = []
    gini = []
    shortage_infos = []  # store clearing_stats per step (use first agent's info per step)
    for t in range(max_steps):
        actions = {}
        for aid, o in obs.items():
            role = env.params[aid]["role"]
            if role == "consumer":
                actions[aid] = consumer_heuristic(o, env)
            elif role == "producer":
                actions[aid] = producer_heuristic(o, env)
            else:
                actions[aid] = trader_heuristic(o, env)
        obs, rewards, dones, infos = env.step(actions)
        # env returns infos per agent; pick first agent info as representative (clearing_stats are same)
        sample_info = next(iter(infos.values()))
        clearing = sample_info.get("clearing_stats", {})
        last_price = sample_info.get("last_price", env.get_last_price())
        prices.append(last_price)
        matched = clearing.get("matched_qty", 0)
        trade_val = matched * last_price
        trade_values.append(trade_val)
        trade_vol.append(matched)
        # compute gini
        wealths = [env.cash[a] + env.inv[a] * last_price for a in env.agents]
        gini.append(gini_coeff(wealths))
        shortage_infos.append(sample_info)
        if all(dones.values()):
            break
    # compute shortage ratio series using clearing_stats from shortage_infos
    shortage_ratio = compute_shortage_ratio_by_step(shortage_infos)
    return {
        "prices": prices,
        "trade_values": trade_values,
        "trade_vol": trade_vol,
        "gini": gini,
        "shortage": shortage_ratio
    }

def build_agents(n_consumers=5, n_producers=5, n_traders=2, seed=0):
    rng = np.random.default_rng(seed)
    configs = []
    for i in range(n_consumers):
        configs.append({
            "id": f"consumer_{i}",
            "role": "consumer",
            "init_cash": float(rng.uniform(60, 140)),
            "init_inv": int(rng.integers(0, 3)),
            "risk_aversion": float(rng.uniform(0.8, 1.2)),
            "consume_cap": int(rng.integers(1, 3))
        })
    for i in range(n_producers):
        configs.append({
            "id": f"producer_{i}",
            "role": "producer",
            "init_cash": float(rng.uniform(40, 120)),
            "init_inv": int(rng.integers(1, 6)),
            "prod_cost": float(rng.uniform(1.5, 3.5)),
            "prod_rate": float(rng.uniform(0.5, 1.2)),
            "risk_aversion": float(rng.uniform(0.9, 1.2))
        })
    for i in range(n_traders):
        configs.append({
            "id": f"trader_{i}",
            "role": "trader",
            "init_cash": float(rng.uniform(100, 200)),
            "init_inv": int(rng.integers(0, 4)),
            "risk_aversion": float(rng.uniform(0.8, 1.5))
        })
    return configs

def run_experiments(outdir: str = "exp_outputs", seeds: List[int] = [0,1,2], max_steps: int = 300):
    os.makedirs(outdir, exist_ok=True)
    # define policies to compare
    policies = {
        "baseline": {"price_cap": None, "price_floor": None, "tax": 0.0, "subsidy": 0.0, "interest_rate": 0.0, "shock_prob": 0.03, "shock_mag": 0.6},
        "cap_3": {"price_cap": 3.0, "price_floor": None, "tax": 0.0, "subsidy": 0.0, "interest_rate": 0.0, "shock_prob": 0.03, "shock_mag": 0.6},
        "tax_2pct": {"price_cap": None, "price_floor": None, "tax": 0.02, "subsidy": 0.0, "interest_rate": 0.0, "shock_prob": 0.03, "shock_mag": 0.6},
    }

    results = {}
    for pname, pparams in policies.items():
        print("Running policy:", pname)
        # collect lists of runs
        prices_runs = []
        tradeval_runs = []
        tradevol_runs = []
        gini_runs = []
        shortage_runs = []
        for s in seeds:
            agent_cfgs = build_agents(seed=s)
            run_data = single_run(agent_cfgs, pparams, seed=s, max_steps=max_steps)
            prices_runs.append(run_data["prices"])
            tradeval_runs.append(run_data["trade_values"])
            tradevol_runs.append(run_data["trade_vol"])
            gini_runs.append(run_data["gini"])
            shortage_runs.append(run_data["shortage"])
        # pad runs to same length (if needed)
        T = max(len(r) for r in prices_runs)
        def pad(rs):
            arr = []
            for r in rs:
                if len(r) < T:
                    # pad with last value
                    pad_val = r[-1] if len(r) > 0 else 0.0
                    arr.append(r + [pad_val] * (T - len(r)))
                else:
                    arr.append(r[:T])
            return arr
        prices_runs = pad(prices_runs)
        tradeval_runs = pad(tradeval_runs)
        tradevol_runs = pad(tradevol_runs)
        gini_runs = pad(gini_runs)
        shortage_runs = pad(shortage_runs)

        results[pname] = {
            "prices": prices_runs,
            "gdp": tradeval_runs,
            "trade_vol": tradevol_runs,
            "gini": gini_runs,
            "shortage": shortage_runs
        }
        # save raw per-policy
        np.savez_compressed(f"{outdir}/{pname}_raw.npz", prices=prices_runs, gdp=tradeval_runs, trade_vol=tradevol_runs, gini=gini_runs, shortage=shortage_runs, seeds=seeds)
    # save a summary json of policies and seeds
    with open(f"{outdir}/meta.json", "w") as f:
        json.dump({"policies": list(policies.keys()), "seeds": seeds, "max_steps": max_steps}, f, indent=2)
    return results

if __name__ == "__main__":
    res = run_experiments(outdir="exp_outputs", seeds=[0,1,2,3,4], max_steps=300)
    # produce comparison plot
    from multi_agent_econ.metrics import comparison_panel_plot
    fpath = comparison_panel_plot(res, outdir="exp_outputs")
    print("Saved comparison panel to:", fpath)
