# coding=utf-8
class Position(object):
    active = False
    entry_signal = ""
    entry_price = None
    entry_bar = None
    exit_bar = None
    exit_commission = 0
    exit_price = None
    exit_signal = ""
    is_long = True
    share_size = 1
    net_profit = 0

    def __init__(self, entry_bar=0, entry_price=0, is_long=True, shares=1):
        self.entry_price = entry_price
        self.entry_bar = entry_bar
        self.active = True
        self.is_long = is_long
        self.share_size = shares
        return

    def calc_current_profit(self, current_close_cost):
        """ расчитывает текущую прибыль позиции"""

        _diff = current_close_cost - self.entry_price
        return _diff*[-1.0, 1.0][self.is_long]
