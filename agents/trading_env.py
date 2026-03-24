import numpy as np
import gymnasium as gym
from gymnasium import spaces
from risk.risk_manager import RiskManager


class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=100000):
        super(TradingEnv, self).__init__()

        # 🔥 FIX: flatten columns if multi-index
        if hasattr(data.columns, "levels"):
            data.columns = data.columns.get_level_values(0)

        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance

        self.current_step = 0

        # Portfolio
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance

        self.risk_manager = RiskManager()
        self.returns = []

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance

        self.risk_manager = RiskManager()
        self.returns = []

        return self._get_observation(), {}

    def _safe_value(self, val):
        """🔥 Extract scalar safely from Series/list"""
        if isinstance(val, (list, np.ndarray)):
            return float(val[0])
        if hasattr(val, "iloc"):
            return float(val.iloc[0])
        return float(val)

    def step(self, action):

        prev_net_worth = self.net_worth

        row = self.data.iloc[self.current_step]
        current_price = self._safe_value(row["Close"])

        # Risk management
        action = self.risk_manager.apply_risk(
            action,
            current_price,
            self.balance,
            self.shares_held,
            self.net_worth
        )

        # BUY
        if action == 1:
            if self.balance > 0:
                self.shares_held = self.balance / current_price
                self.balance = 0
                self.risk_manager.entry_price = current_price

        # SELL
        elif action == 2:
            if self.shares_held > 0:
                self.balance = self.shares_held * current_price
                self.shares_held = 0
                self.risk_manager.entry_price = None

        self.net_worth = self.balance + self.shares_held * current_price

        # Reward
        log_return = np.log(self.net_worth / prev_net_worth) if prev_net_worth > 0 else 0
        self.returns.append(log_return)

        volatility = np.std(self.returns) if len(self.returns) > 1 else 0
        reward = log_return - (0.1 * volatility)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_observation(), reward, done, False, {}

    def _get_observation(self):
        row = self.data.iloc[self.current_step]

        obs = np.array([
            self._safe_value(row["Close"]),
            self._safe_value(row["rsi"]),
            self._safe_value(row["macd"]),
            self._safe_value(row["macd_signal"]),
            self._safe_value(row["sentiment"]),
            self._safe_value(row["regime"]),
            float(self.balance),
            float(self.shares_held)
        ], dtype=np.float32)

        return obs