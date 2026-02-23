# demo.py
from econ_env import EconEnv
from agents import consumer_heuristic, producer_heuristic, trader_heuristic
from metrics import plot_price_series, plot_wealth_hist, gini_coeff, plot_gini_over_time
import numpy as np
import os
import matplotlib.pyplot as plt

def build_agents(n_consumers=5, n_producers=5, n_traders=2, seed=0):
    rng = np.random.default_rng(seed)
    configs = []
    # Consumers (heterogeneous wealth & consumption cap)
    for i in range(n_consumers):
        configs.append({
            "id": f"consumer_{i}",
            "role": "consumer",
            "init_cash": float(rng.uniform(60, 140)),
            "init_inv": int(rng.integers(0, 3)),
            "risk_aversion": float(rng.uniform(0.8, 1.2)),
            "consume_cap": int(rng.integers(1, 3))
        })
    # Producers
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
    # Traders
    for i in range(n_traders):
        configs.append({
            "id": f"trader_{i}",
            "role": "trader",
            "init_cash": float(rng.uniform(100, 200)),
            "init_inv": int(rng.integers(0, 4)),
            "risk_aversion": float(rng.uniform(0.8, 1.5))
        })
    return configs

def run_experiment(policy_params, seed=0, fname_prefix="run", show_plots=False):
    agent_configs = build_agents(seed=seed)
    env = EconEnv(agent_configs, max_steps=300, price_grid=list(range(1, 11)),
                  policy_params=policy_params, seed=seed,
                  reward_config={"type": "wealth_plus_consumption", "consumption_weight": 1.0, "inv_penalty": 0.0})
    obs, _ = env.reset()
    gini_over_time = []
    price_series = []
    for t in range(env.max_steps):
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
        # record gini
        last_price = env.get_last_price()
        wealths = [env.cash[a] + env.inv[a] * last_price for a in env.agents]
        gini_over_time.append(gini_coeff(wealths))
        price_series.append(last_price)
        if all(dones.values()):
            break

    final_price_series = env.price_series
    final_wealths = [env.cash[a] + env.inv[a] * env.get_last_price() for a in env.agents]
    return {
        "price_series": final_price_series,
        "wealths": final_wealths,
        "gini_over_time": gini_over_time,
        "env": env
    }

def main():
    outdir = "demo_outputs"
    os.makedirs(outdir, exist_ok=True)

    # Baseline
    baseline_params = {"price_cap": None, "price_floor": None, "tax": 0.0, "subsidy": 0.0, "interest_rate": 0.0, "shock_prob": 0.03, "shock_mag": 0.6}
    print("Running baseline with stochastic production/consumption...")
    baseline = run_experiment(baseline_params, seed=2, fname_prefix="baseline")

    # Price cap experiment
    cap_params = {"price_cap": 3.0, "price_floor": None, "tax": 0.0, "subsidy": 0.0, "interest_rate": 0.0, "shock_prob": 0.03, "shock_mag": 0.6}
    print("Running price-cap experiment (cap=3.0)...")
    cap_run = run_experiment(cap_params, seed=2, fname_prefix="cap3")

    # Price series compare
    plot_price_series([baseline["price_series"], cap_run["price_series"]], labels=["baseline", "cap=3"], fname=f"{outdir}/price_compare.png")
    print(f"Saved {outdir}/price_compare.png")

    # Wealth histograms
    plot_wealth_hist(baseline["wealths"], fname=f"{outdir}/wealth_baseline.png")
    plot_wealth_hist(cap_run["wealths"], fname=f"{outdir}/wealth_cap.png")
    print(f"Saved wealth histograms in {outdir}/")

    # Gini compare (plot both in one figure)
    plt.figure(figsize=(7,3))
    plt.plot(baseline["gini_over_time"], label="baseline")
    plt.plot(cap_run["gini_over_time"], label="cap=3")
    plt.xlabel("Step")
    plt.ylabel("Gini")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{outdir}/gini_compare.png", dpi=200)
    plt.close()
    print(f"Saved {outdir}/gini_compare.png")

    print("Baseline final Gini:", baseline["gini_over_time"][-1])
    print("Cap final Gini:", cap_run["gini_over_time"][-1])
    print("Demo completed. Plots saved in demo_outputs/")

if __name__ == "__main__":
    main()
