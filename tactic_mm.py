from utils import Min, Hour


class TacticMM:
    def __init__(self):
        pass

    def handle_candles(self, candles1s, current_time):
        t = current_time
        c1h = candles1s.get_candles(Hour().to_sec(1), t - Hour().to_sec(24), t)
        c15m = candles1s.get_candles(Min().to_sec(15), t - Hour().to_sec(6), t)
        c1m = candles1s.get_candles(Min().to_sec(1), t - Min().to_sec(24), t)

        c1h.printf()
        c15m.printf()
        c1m.printf()
