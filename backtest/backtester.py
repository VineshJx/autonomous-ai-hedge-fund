import numpy as np
from agents.trading_env import TradingEnv


class Backtester:
    def __init__(self, data, model):
        self.env = TradingEnv(data)
        self.model = model

    def run(self):

        obs, _ = self.env.reset()
        done = False

        portfolio_values = []

        while not done:
            action, _ = self.model.predict(obs)

            obs, reward, done, truncated, info = self.env.step(action)

            portfolio_values.append(self.env.net_worth)

            if truncated:
                break

        portfolio_values = np.array(portfolio_values)

        final_value = portfolio_values[-1]
        total_return = (final_value / portfolio_values[0]) - 1

        returns = np.diff(portfolio_values) / portfolio_values[:-1]

        sharpe_ratio = (
            np.mean(returns) / np.std(returns)
            if np.std(returns) != 0
            else 0
        )

        max_drawdown = np.max(
            np.maximum.accumulate(portfolio_values) - portfolio_values
        ) / np.max(portfolio_values)

        return {
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }