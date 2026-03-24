class RiskManager:
    def __init__(
        self,
        max_position_size=0.3,   # max 30% capital per trade
        stop_loss_pct=0.05,      # 5% stop loss
        max_drawdown_pct=0.2     # 20% max drawdown
    ):
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        self.entry_price = None
        self.peak_net_worth = None

    def update_peak(self, net_worth):
        if self.peak_net_worth is None:
            self.peak_net_worth = net_worth
        else:
            self.peak_net_worth = max(self.peak_net_worth, net_worth)

    def check_drawdown(self, net_worth):
        if self.peak_net_worth is None:
            return False

        drawdown = (self.peak_net_worth - net_worth) / self.peak_net_worth
        return drawdown > self.max_drawdown_pct

    def apply_risk(self, action, price, balance, shares_held, net_worth):

        self.update_peak(net_worth)

        # 🔴 Max drawdown protection
        if self.check_drawdown(net_worth):
            return 2  # force SELL

        # 🔴 Stop loss
        if shares_held > 0 and self.entry_price is not None:
            loss = (price - self.entry_price) / self.entry_price
            if loss < -self.stop_loss_pct:
                return 2  # force SELL

        # 🟢 Position sizing
        if action == 1:  # BUY
            max_buy_amount = balance * self.max_position_size
            if max_buy_amount < balance:
                return 1  # controlled buy

        return action