class SimSummary:
    def __init__(self, initial_xbt=None, position_xbt=None, num_fills=None, num_orders=None, num_cancels=None,
                 num_liq=None, close_price=None,
                 pnl=None, pnl_total=None, profit_total=None, loss_total=None):
        self.initial_xbt = initial_xbt  # type: float
        self.position_xbt = position_xbt  # type: float
        self.num_fills = num_fills  # type: dict
        self.num_orders = num_orders  # type: dict
        self.num_cancels = num_cancels  # type: dict
        self.num_liq = num_liq  # type: dict
        self.close_price = close_price  # type: float  # TODO: should be dict
        self.pnl = pnl  # type: dict
        self.pnl_total = pnl_total  # type: float
        self.profit_total = profit_total  # type: float
        self.loss_total = loss_total  # type: float

    def to_str(self):
        li = ["initial_xbt",
              "position_xbt",
              "num_fills",
              "num_orders",
              "num_cancels",
              "num_liq",
              "close_price",
              "pnl",
              "pnl_total",
              "profit_total",
              "loss_total"]

        s = ""
        for ll in li:
            s += "{}: {}\n".format(ll, str(getattr(self, ll)))
        return s
