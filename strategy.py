#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import time
import position


class Strategy(object):
    """strategy template"""

    def __init__(self, bars):
        self.bars = bars
        self.bars_count = len(bars)
        self.active_positions = {}
        self.closed_positions = {}

        self.end_of_day = time(23, 45)
        self.drowdown = 0.  # просадка
        self.max_drowdown = 0.
        self.net_profit = 0.
        self.profit_closed_pos = 0.
        self.profit_active_pos = 0.
        self.sensitivity = 0.8
        self.share_size = 1
        self.starting_capital = 0.
        self.total_money = 0.
        self.penalty = 0.0001
        self.step_cost = 0.0001
        self.comission = 1
        self.bar = 0
        self.lastStepProfit = 0
        self.State_values = (0, 0, 0)  # (long_pos, short_pos, last profit)

        self.multipose = False  # возможность одновременно держать несколько отркрытых позиций

        return

    def reset(self):
        self.closed_positions = {}
        self.active_positions = {}
        self.net_profit = 0.
        self.total_money = 0.
        self.max_drowdown = 0.
        self.drowdown = 0.
        self.bar = 0
        self.profit_active_pos = 0.0
        self.profit_closed_pos = 0.0
        return

    def calc_profit_by_active_pos(self):
        """ расчитывает текущую прибыль по всем активным позициям"""

        _profit = 0.0
        _close_cost = self.bars.Close.values[self.bar]
        for _pos_key in self.active_positions:
            _profit = _profit + self.active_positions[_pos_key].calc_current_profit(_close_cost) - self.comission
        return _profit

    def calc_total_profit(self):
        """
        рассчитаваем профит по всем закрытым и активным позициям
        """
        _old_profit = self.net_profit
        self.profit_active_pos = self.calc_profit_by_active_pos()
        self.net_profit = self.profit_active_pos + self.profit_closed_pos
        self.net_profit -= self.bar * self.step_cost
        self.lastStepProfit = self.net_profit - _old_profit
        return self.net_profit

    def open_at_market(self, bar, islong, shares=1):
        p = position.Position(entry_bar=bar, entry_price=self.bars.Open.values[bar], is_long=islong, shares=shares)

        self.active_positions[hash(p)] = p

        return

    def close_at_market(self, bar, pos_key):
        ''' Закрытие единичной позиции и перенос ее в архивные'''
        _close_cost = self.bars.Open.values[self.bar]
        _profit = self.active_positions[pos_key].calc_current_profit(_close_cost)
        _cp = self.active_positions.pop(pos_key)
        _cp.active = False
        _cp.exit_bar = self.bar
        _cp.exit_price = _close_cost
        _cp.net_profit = _profit
        _cp.exit_signal = "CL_M"
        # self.closed_positions[pos_key] = _cp
        self.profit_closed_pos += _profit - self.comission
        self.profit_active_pos -= _profit

        return

    def total_active_shares(self):
        _total_shares = 0
        for pos in self.active_positions:
            _total_shares += self.active_positions[pos].share_size * ([-1, 1][self.active_positions[pos].is_long])

        return _total_shares

    def close_all_active_pos(self):
        _pos_keys = list(self.active_positions.keys())
        for _pos_key in _pos_keys:
            self.close_at_market(self.bar, _pos_key)
        return

    def next_bar(self, action=0):
        """
        action -Действия
        0 - не делать ничего
        1 - закрыть все  позиции
        2 - реверс позицию
        3 - лонг по рынку
        4 - шорт по рынку
        Если находимся на предпоследнем баре - закрываем все активные позиции
        TODO Если находимся на последнем баре - выдаем ошибку
        """
        if self.bar >= self.bars_count - 1:  # если мы находимся уже на последнем баре, то ничего не делаем
            return

        if self.bar == self.bars_count - 2:  # если текущий бар предпоследний то заменяем действие на закрытие бара
            action = 1

        self.bar += 1
        self.do_action_on_bar(bar=self.bar, action=action)

        return

    def do_action_on_bar(self, bar, action=0):
        """
        Выполняет действие на указанном баре
        action -Действия
        0 - не делать ничего
        1 - закрыть все  позиции
        2 - реверс позицию
        3 - лонг по рынку
        4 - шорт по рынку

        Новые
        0 - закрыть все позиции, не делать ничего

        1 - лонг по рынку - закрыть все шорты открыть лонги
        2 - шорт по рынку - закрыть все лонги открыть шорты



        Если находимся на предпоследнем баре - закрываем все активные позиции
        TODO Если находимся на последнем баре - выдаем ошибку
        """

        if self.bar == self.bars_count - 1:  # если текущий  последний, то единственное, что мы можем стделать, это попытаться закрыть все активные позиции
            action=0
        elif self.bar > self.bars_count - 1:
            return
        # ------------------ не делаем ничего -------------
        if action == 10:
            self.calc_total_profit()
            return
        # ------------------ закрываем активные позиции -------------
        if action == 0:
            self.close_all_active_pos()
            self.calc_total_profit()
            self.State_values=(0,0,self.lastStepProfit)
            return

        # -----------реверс позицию-------------------
        if action == 11:
            _total_shares = self.total_active_shares()
            self.close_all_active_pos()
            # Открываем эквивалентную позицию в противоположном направлении
            if abs(_total_shares) > 0:
                self.open_at_market(bar=bar, islong=_total_shares < 0, shares=abs(_total_shares))

            self.calc_total_profit()
            return

        if action == 1:  # если есть, закрыть короткие, открыть длинные
            _total_shares = self.total_active_shares()
            if _total_shares < 0:  # если есть короткие то все закрыть
                self.close_all_active_pos()
                self.open_at_market(bar=bar, islong=True, shares=self.share_size)
            elif _total_shares == 0:  # если нет позиций
                self.open_at_market(bar=bar, islong=True, shares=self.share_size)
            self.calc_total_profit()

            self.State_values = (1, 0, self.lastStepProfit)
            return

        if action == 2:  # если есть, закрыть длинные, открыть корокие
            _total_shares = self.total_active_shares()
            if _total_shares > 0:  # если есть короткие то все закрыть
                self.close_all_active_pos()
                self.open_at_market(bar=bar, islong=False, shares=self.share_size)
            elif _total_shares == 0:  # если нет позиций
                self.open_at_market(bar=bar, islong=False, shares=self.share_size)
            self.calc_total_profit()
            self.State_values = (0, 1, self.lastStepProfit)
            return

        return

    def predict_best_next_action(self, bar):
        _best_action = 0
        if self.bar >= self.bars_count - 1:
            return 0

        _future_dela = self.bars.Close.values[bar] - self.bars.Open.values[bar]

        if _future_dela > self.comission:
            return 1
        elif _future_dela > -self.comission:
            return 2
        else:
            return 0

        return _best_action
