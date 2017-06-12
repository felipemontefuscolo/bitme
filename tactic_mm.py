class TacticMM:
    def __init__(self):
        pass

    def handle_candles(self, candles1s):
        t = candles1s.ts_l[-1]  # current time
        c1m = candles1s.get_candl
