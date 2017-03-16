# coding=utf-8
import dataIO
import strategy as strat
import copy as cpy
import numpy as np
import random
import pandas as pd

class TradeBot:
    def __init__(self):
        self.current_day_step = 0  # текущий индекс внутри дня
        self.current_day = 0  # текущий день
        self.current_global_step = 0  # сквозной индекс по всей базе
        self.DaysCount = 0
        self.score = 0
        self.global_index = []
        self.money = 0
        self.check_points = {}
        self.check_point_ind = -1
        self.score = 0.0
        self.score_without_curr_strat = 0.0
        self.score_by_curr_strat = 0.0

        self.Data = None

        self.days_steps = None
        self.total_steps = None
        self.current_day_vectors = None
        self.current_day_candles = None
        self.strategy = None
        self.check_points = {}
        self.level_finished = False
        self.level_loaded = False

        return

    def load_level_LSTM(self, filename, verbose=True, day_len=-1,time_steps=12):
        """

            Параметры:
            filename: str Путь до файла с уровнем.
            verbose: bool
        """
        if verbose:
            print(u'Открываем файл ', filename)
        self.Arch_Data=dataIO.load_for_LSTM_Bot_by_Dates(filename, time_steps=6)
        self.Data = self.Arch_Data
        self.DaysCount = len(self.Data)
        if day_len != -1:
            self.apply_data_len(len=day_len)

        self.days_steps = [len(day[1]) for day in self.Data]
        self.total_steps = sum(self.days_steps)
        self.current_day_vectors = self.Data[self.current_day]
        self.current_day_candles = self.Data[self.current_day][2]
        self.strategy = strat.Strategy(self.current_day_candles)
        self.check_point_ind = -1
        self.check_points = {}
        self.level_finished = False
        self.global_index = []
        for i in range(len(self.Data)):

            for j in range(len(self.Data[i][1])):
                self.global_index.append((i, j))
        self.level_loaded = True
        if verbose:
            print('Done: ', self.DaysCount, " days")

        return

    def load_by_Dates(self, filename):
        '''загрузка данных без попытки  группировать по временным шагам'''
        data = np.load(filename)

        c = pd.DataFrame(data.f.candles, columns=data.f.candles_names)
        # TODO начисто забыл что такое deltas
        X = np.hstack((data.f.vectors, data.f.deltas))

        # categorical_cols = ['Qty', 'hour', 'minute']
        # X = np.hstack((X, c[categorical_cols].applymap(int).values))

        dates = list(set(c["DateTime"].apply(lambda x: x.date())))
        # print (dates)  #TODO проверить тут.
        d = sorted(dates)
        # print(d)  # TODO проверить тут.
        data_byDates = []

        for d1 in d:
            date_indx = c["DateTime"].apply(lambda x: x.date()) == d1
            x_d = X[date_indx.index[date_indx]]
            c_d = c[date_indx]

            data_byDates.append([d1, x_d, c_d])

        return data_byDates


    def load_level_from_Data(self, data ):
        """
                    загружаем предобработанные данные, сгрупированные по дням.
                    Параметры:
                    filename: str Путь до файла с уровнем.
                    verbose: bool
                """

        self.Arch_Data = data
        self.Data = self.Arch_Data
        self.DaysCount = len(self.Data)

        self.days_steps = [len(day[1]) for day in self.Data]
        self.total_steps = sum(self.days_steps)
        self.current_day_vectors = self.Data[self.current_day]
        self.current_day_candles = self.Data[self.current_day][2]
        self.strategy = strat.Strategy(self.current_day_candles)
        self.check_point_ind = -1
        self.check_points = {}
        self.level_finished = False
        self.global_index = []
        for i in range(len(self.Data)):

            for j in range(len(self.Data[i][1])):
                self.global_index.append((i, j))
        self.level_loaded = True

        return



    def load_level(self, filename, verbose=True, day_len=-1):
        """
            Функция, загружающая файл с уровнем. Уровень должен быть загружен до начала игры.
            Параметры:
            filename: str Путь до файла с уровнем.
            verbose: bool
        """
        if verbose:
            print(u'Открываем файл ', filename)
        self.Arch_Data=dataIO.load_by_Dates(filename)
        self.Data = self.Arch_Data
        self.DaysCount = len(self.Data)
        if day_len != -1:
            self.apply_data_len(len=day_len)

        self.days_steps = [len(day[1]) for day in self.Data]
        self.total_steps = sum(self.days_steps)
        self.current_day_vectors = self.Data[self.current_day]
        self.current_day_candles = self.Data[self.current_day][2]
        self.strategy = strat.Strategy(self.current_day_candles)
        self.check_point_ind = -1
        self.check_points = {}
        self.level_finished = False
        self.global_index = []
        for i in range(len(self.Data)):

            for j in range(len(self.Data[i][1])):
                self.global_index.append((i, j))
        self.level_loaded = True
        if verbose:
            print('Done: ', self.DaysCount, " days")

        return

    def set_sample_days(self, days_to_sample=1):
        self.Data = random.sample(self.Arch_Data, days_to_sample)
        self.DaysCount = len(self.Data)


        self.days_steps = [len(day[1]) for day in self.Data]
        self.total_steps = sum(self.days_steps)
        self.current_day_vectors = self.Data[self.current_day]
        self.current_day_candles = self.Data[self.current_day][2]
        self.strategy = strat.Strategy(self.current_day_candles)
        self.check_point_ind = -1
        self.check_points = {}
        self.level_finished = False
        self.global_index = []
        for i in range(len(self.Data)):

            for j in range(len(self.Data[i][1])):
                self.global_index.append((i, j))
        self.level_loaded = True


        return



    def is_level_loaded(self):
        return self.level_loaded

    def apply_data_len(self, len=5):
        for day in range(self.DaysCount):
            for data_item in range(1, np.shape(self.Data[day])[0]):
                self.Data[day][data_item] = self.Data[day][data_item][:len]

    def get_state(self):
        """Возвращает numpy.float32-массив с вектором текущего состояния игры"""
        _st=None
        try:
                _st = self.Data[self.current_day][1][self.current_day_step].flatten()
        except:
            print (self.current_day, self.current_day_step )
       # _strategyState=np.array(self.strategy.State_values)
       # print(_strategyState)
        #_st =np.hstack((_st, _strategyState))
        return  _st#_st.flatten()

    def get_lstm_state(self):
        return self.Data[self.current_day][1][self.current_day_step]

    def get_score(self):
        """
        Суммарная прибыль с начала симуляции.
        """

        return self.score

    def reset_level(self):
        '''
        Сбрасываем бота в начальне состояние.
        '''
        self.current_day_step = 0
        self.current_day = 0
        self.current_global_step = 0
        self.current_day_candles = self.Data[self.current_day][2]
        self.strategy = strat.Strategy(self.current_day_candles)
        self.score = 0.0
        self.score_by_curr_strat = 0.0
        self.score_without_curr_strat = 0.0
        self.level_finished = False
        return

    def do_action(self, action):
        """ Агент сделает действие action, и игра перейдёт к следущему шагу. Так как агенту доступно 3 действия, action
        должно принимать целое значение от 0 до 3.
        Параметры:
        action: int
        Действие, которое сделает агент на текущем шаге.
        Действия

        Новые
        0 - не делать ничего

        3 - лонг по рынку - закрыть все шорты открыть лонги
        4 - шорт по рынку - закрыть все лонги открыть шорты

        """
        #если уровень уже закончен, то не делаем ничего, возвращаем false
        if self.level_finished:
            return False
        self.next()  # прокручиваем все счетчики на шаг вперед, в том числе стратегию двигаем на следующую, если надо

        if self.level_finished:
            return False
        self.strategy.do_action_on_bar(bar=self.current_day_step, action=action)
        self.score_by_curr_strat = self.strategy.net_profit

        self.score = self.score_without_curr_strat + self.score_by_curr_strat  # обновляем профит по стратегии
        return True

    def next(self):
        """
        сдвинуть все счетчики на 1
        если переместились в след. день - поменять стратегию
        если находимся в последнем баре, никуда не смещаемся
        """
        if self.current_global_step >= self.total_steps - 1:  # никуда не смещаемся, если дошли до последнего бара
            self.level_finished = True
            return
        _before_day = self.current_day  # запоминаем текущий день
        self.current_global_step += 1  # увеличиваем глобальный шаг на день
        # обновляем счетики дня  счетчик бара внутри дня
        self.current_day, self.current_day_step = self.get_day_step(self.current_global_step)

        # если мы переместились в следующий день, то надо поменять стратегию на след день
        if _before_day < self.current_day:
            self.score_without_curr_strat += self.score_by_curr_strat
            self.score_by_curr_strat = 0.0
            self.current_day_candles = self.Data[self.current_day][2]
            self.strategy = strat.Strategy(self.current_day_candles)  # создаем новую стратегию
            # print "next day : ", self.current_day
            # так как стратегия поменялась - надо о обновить профиты
        else:
            self.strategy.bar += 1

        return

    def get_day_step(self, global_step):
        """
        вычисляет текущий день и текущий индекс
        """

        try:
            _day = self.global_index[global_step][0]
        except:
            print("неправильный индекс: ", global_step)
            _day = self.global_index[-1][0]
        _idx = self.global_index[global_step][1]
        return _day, _idx

    def get_num_of_features(self):
        """Возвращает длину вектора, описывающего состояние среды. Длина всегда фиксированна."""

        return np.product(self.Data[0][1][0].shape)#+3
    def get_num_of_lstm_features(self):
        return self.Data[0][1][0].shape

    def make_point(self):
        _strat = cpy.copy(self.strategy)
        _strat.active_positions = cpy.copy(self.strategy.active_positions)
        _strat.closed_positions = cpy.copy(self.strategy.closed_positions)
        _point = (self.current_global_step,
                  self.current_day,
                  self.current_day_step,
                  self.score,
                  self.score_by_curr_strat,
                  self.score_without_curr_strat,
                  self.level_finished,
                  _strat)

        return _point

    def restore_from_checkpoint(self, point):
        self.current_global_step, \
        self.current_day, \
        self.current_day_step, \
        self.score, \
        self.score_by_curr_strat, \
        self.score_without_curr_strat, \
        self.level_finished = (point[:len(point) - 1])
        _strategy = cpy.copy(point[-1])
        self.strategy = _strategy
        self.strategy.active_positions = cpy.copy(_strategy.active_positions)
        self.strategy.closed_positions = cpy.copy(_strategy.closed_positions)

        return

    def create_checkpoint(self):
        """
         Создает чекпоинт с которого можно загрузиться позднее. Возвращает уникальный номер чекпоинта. Номера чекпоинтов монотонно возрастают.
         Примечание: при вызове функции load_level(..) все созданные чекпоинты удаляются. Вызов функции reset_level() сохраняет чекпоинты.

         1.Сохранить текущие номера шагов по циклу.
         2Сделать копии стратегий

        """
        self.check_point_ind += 1
        self.check_points[self.check_point_ind] = self.make_point()
        return self.check_point_ind

    def load_from_checkpoint(self, checkpoint_id):

        """
            Возвращает игру в состояние, в котором был создан checkpoint_id. Параметры:
            checkpoint_id: int номер чекпоинта для загрузки
        """
        self.restore_from_checkpoint(self.check_points[checkpoint_id])
        return

    def get_num_of_checkpoints(self):
        """
        Возвращает число созданных чекпоинтов.
        """
        return self.check_point_ind

    def clear_all_checkpoints(self):
        """
        Удаляет все чекпоинты.
        """
        self.check_point_ind = -1
        self.check_points = {}
        return

    def get_max_time(self):
        """
        Возвращает число шагов в уровне. Число шагов зафиксированно для одного уровня, но может меняться от уровня к уровню.
        :return: INT
        """
        return self.total_steps

    def is_level_finished(self):
        """
        Возвращает True, если текущий уровень закончен.
        :return:
        """
        return self.level_finished

    def finish(self, verbose=True):
        """
        Эту функцию нужно вызывать в конце игровой симуляции.
        Возвращает число набранных очков в конце симуляции.
        Параметры:
        verbose: bool
        Включает печать количества очков в консоль..
        :param verbose: 
        :return:
        """
        if verbose:
            print
            self.score
        return self.score

    def get_num_of_actions(self):
        return 3

    def get_time(self):
        """
        Возвращает "время" — номер текущего шага в игре. Каждый уровень всегда длится фиксированное число шагов.
        """
        return self.current_global_step
