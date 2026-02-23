# econ_env.py
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from .market import clear_market

AgentID = str
Order = Dict[str, Any]
Trade = Dict[str, Any]

class EconEnv:
    def __init__(
        self,
        agent_configs: List[Dict],
        max_steps: int = 300,
        price_grid: Optional[List[float]] = None,
        max_qty: int = 3,
        clearing_method: str = "batch",
        policy_params: Optional[Dict] = None,
        reward_config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.t = 0

        # Agents and their configs
        self.agent_configs = {cfg["id"]: cfg for cfg in agent_configs}
        self.agents = list(self.agent_configs.keys())

        self.cash = {aid: cfg.get("init_cash", 100.0) for aid, cfg in self.agent_configs.items()}
        self.inv = {aid: cfg.get("init_inv", 0.0) for aid, cfg in self.agent_configs.items()}
        self.params = {aid: cfg for aid, cfg in self.agent_configs.items()}

        self.price_grid = price_grid or list(np.linspace(1, 10, 10))
        self.max_qty = max_qty
        self.clearing_method = clearing_method

        self.policy_params = policy_params or {
            "price_cap": None,
            "price_floor": None,
            "tax": 0.0,
            "subsidy": 0.0,
            "interest_rate": 0.0,
            "shock_prob": 0.0,
            "shock_mag": 1.0,
        }

        # reward_config can include consumer utility weight, inventory penalty weights, etc.
        self.reward_config = reward_config or {"type": "wealth_plus_consumption", "consumption_weight": 1.0}

        # logs
        self.price_series: List[float] = []
        self.trade_history: List[Trade] = []
        self.trade_volume_series: List[float] = []
        self.wealth_history: List[List[float]] = []

        self._prev_wealth = {aid: self.cash[aid] + self.inv[aid] * self.get_last_price() for aid in self.agents}
        # simple EMA trackers
        self.ema_demand = 1.0
        self.ema_supply = 1.0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        for aid, cfg in self.agent_configs.items():
            self.cash[aid] = cfg.get("init_cash", 100.0)
            self.inv[aid] = cfg.get("init_inv", 0.0)
        self.price_series = []
        self.trade_history = []
        self.trade_volume_series = []
        self.wealth_history = []
        self._prev_wealth = {aid: self.cash[aid] + self.inv[aid] * self.get_last_price() for aid in self.agents}
        self.ema_demand = 1.0
        self.ema_supply = 1.0
        obs = {aid: self._make_observation(aid) for aid in self.agents}
        infos = {aid: {} for aid in self.agents}
        return obs, infos

    def step(self, actions: Dict[AgentID, Tuple[int, int, int]]):
        # 1) interest on cash
        irate = self.policy_params.get("interest_rate", 0.0)
        if irate != 0.0:
            for a in self.agents:
                self.cash[a] *= (1.0 + irate)

        # 2) shock?
        shock_info = {"shock": False, "mag": 1.0}
        if self.rng.random() < self.policy_params.get("shock_prob", 0.0):
            shock_info = {"shock": True, "mag": self.policy_params.get("shock_mag", 1.0)}

        # 3) actions -> orders
        orders: List[Order] = []
        for aid, act in actions.items():
            o = self._action_to_order(aid, act)
            if o is not None:
                orders.append(o)
        # adjust demand/supply EMAs for observations (simple)
        total_bid = sum([o["qty"] for o in orders if o["side"] == "buy"])
        total_ask = sum([o["qty"] for o in orders if o["side"] == "sell"])
        # incorporate shock into demand as multiplicative factor
        self.ema_demand = 0.9 * self.ema_demand + 0.1 * (total_bid * shock_info["mag"] + 1e-6)
        self.ema_supply = 0.9 * self.ema_supply + 0.1 * (total_ask + 1e-6)

        # 4) market clearing
        trades, clearing_price, stats = clear_market(
            orders=orders,
            last_price=self.get_last_price(),
            method=self.clearing_method,
            price_grid=self.price_grid,
            rng=self.rng,
            demand_mult=shock_info["mag"] if shock_info["shock"] else 1.0
        )

        # 5) apply price caps/floors
        settlement_price = clearing_price
        cap = self.policy_params.get("price_cap", None)
        floor = self.policy_params.get("price_floor", None)
        if cap is not None:
            settlement_price = min(settlement_price, cap)
        if floor is not None:
            settlement_price = max(settlement_price, floor)
        settlement_price = max(1e-6, settlement_price)

        # 6) settle trades and update balances
        executed = self._settle_trades(trades, settlement_price)

        # 7) Production (producers produce goods stochastically) and consumption (consumers consume)
        produced = {}
        consumed = {}
        for a in self.agents:
            role = self.params[a].get("role", "")
            if role == "producer":
                # Poisson production with mean prod_rate (agent-specific or default)
                prod_rate = self.params[a].get("prod_rate", 0.8)
                qty_prod = int(self.rng.poisson(prod_rate))
                cost_per_unit = self.params[a].get("prod_cost", 2.0)
                total_cost = qty_prod * cost_per_unit
                # producers can invest in production if they have cash (allow negative cash disallowed)
                if total_cost <= self.cash[a]:
                    self.inv[a] += qty_prod
                    self.cash[a] -= total_cost
                else:
                    # produce fewer if can't afford
                    max_aff = int(self.cash[a] // max(1e-6, cost_per_unit))
                    qty_prod = min(qty_prod, max_aff)
                    self.inv[a] += qty_prod
                    self.cash[a] -= qty_prod * cost_per_unit
                produced[a] = qty_prod
            elif role == "consumer":
                # consume 0..consumption_cap units (random)
                cap_consume = self.params[a].get("consume_cap", 1)
                qty_cons = int(self.rng.integers(0, cap_consume + 1))
                qty_cons = min(qty_cons, int(self.inv[a]))  # can't consume more than inventory
                self.inv[a] -= qty_cons
                consumed[a] = qty_cons
            else:
                produced[a] = 0
                consumed[a] = 0

        # 8) logs & time
        self.t += 1
        self.price_series.append(settlement_price)
        total_traded_qty = sum([tr["qty"] for tr in executed])
        self.trade_volume_series.append(total_traded_qty)
        self.trade_history.extend(executed)
        # wealth history snapshot
        self.wealth_history.append([self.cash[a] + self.inv[a] * settlement_price for a in self.agents])

        # 9) compute rewards
        rewards = {}
        current_wealth = {a: self.cash[a] + self.inv[a] * settlement_price for a in self.agents}
        for a in self.agents:
            prev = self._prev_wealth.get(a, current_wealth[a])
            delta_w = float(current_wealth[a] - prev)
            # consumer utility bonus
            utility_bonus = 0.0
            if self.params[a].get("role", "") == "consumer":
                q = consumed.get(a, 0)
                # log utility: log(1+q)
                utility_bonus = self.reward_config.get("consumption_weight", 1.0) * np.log1p(q)
            # optionally penalize large inventories for traders
            inv_penalty = 0.0
            if self.params[a].get("role", "") == "trader":
                lambda_inv = self.reward_config.get("inv_penalty", 0.0)
                inv_penalty = -lambda_inv * (self.inv[a] ** 2)
            rewards[a] = float(delta_w + utility_bonus + inv_penalty)
        self._prev_wealth = current_wealth

        # 10) obs, infos, dones
        obs = {a: self._make_observation(a) for a in self.agents}
        infos = {a: {"executed_trades": [tr for tr in executed if a in (tr["buyer"], tr["seller"])],
                     "last_price": settlement_price,
                     "clearing_stats": stats,
                     "shock": shock_info,
                     "produced": produced.get(a, 0),
                     "consumed": consumed.get(a, 0)}
                 for a in self.agents}
        done = self.t >= self.max_steps
        dones = {a: done for a in self.agents}
        return obs, rewards, dones, infos

    def _make_observation(self, aid: AgentID):
        last_price = self.get_last_price()
        cash = self.cash[aid]
        inv = self.inv[aid]
        time_frac = self.t / max(1, self.max_steps)
        params = self.params[aid]
        risk = params.get("risk_aversion", 1.0)
        # obs vector (un-normalized): you should normalize for NN training
        return np.array([cash, inv, last_price, self.ema_demand, self.ema_supply, time_frac, risk], dtype=float)

    def _action_to_order(self, aid: AgentID, action: Tuple[int, int, int]) -> Optional[Order]:
        # action: (price_idx, qty, side) where side: 0=hold,1=buy,2=sell
        price_idx, qty, side = action
        price_idx = int(np.clip(price_idx, 0, len(self.price_grid) - 1))
        qty = int(np.clip(qty, 0, self.max_qty))
        if side == 0 or qty == 0:
            return None
        price = float(self.price_grid[price_idx])
        if side == 1:
            # buy -> affordability check
            if self.cash[aid] < price * qty:
                max_aff = int(self.cash[aid] // max(1e-6, price))
                if max_aff <= 0:
                    return None
                qty = min(qty, max_aff)
        else:
            # sell -> inventory check
            if self.inv[aid] < qty:
                qty = int(min(qty, self.inv[aid]))
                if qty <= 0:
                    return None
        return {"agent": aid, "side": "buy" if side == 1 else "sell", "price": price, "qty": qty}

    def _settle_trades(self, trades: List[Trade], settlement_price: float) -> List[Trade]:
        executed = []
        tax = self.policy_params.get("tax", 0.0)
        subsidy = self.policy_params.get("subsidy", 0.0)
        for tr in trades:
            buyer = tr["buyer"]; seller = tr["seller"]; qty = tr["qty"]
            trade_val = settlement_price * qty
            tax_amt = tax * trade_val
            sub_amt = subsidy * trade_val
            # safety checks
            if self.cash[buyer] + 1e-9 < (trade_val + tax_amt):
                continue
            if self.inv[seller] + 1e-9 < qty:
                continue
            self.cash[buyer] -= (trade_val + tax_amt)
            self.inv[buyer] += qty
            self.cash[seller] += (trade_val + sub_amt)
            self.inv[seller] -= qty
            executed.append({"buyer": buyer, "seller": seller, "qty": qty, "price": settlement_price})
        return executed

    def get_last_price(self) -> float:
        if len(self.price_series) == 0:
            # default initial price = middle of grid
            return float(self.price_grid[len(self.price_grid) // 2])
        return float(self.price_series[-1])
