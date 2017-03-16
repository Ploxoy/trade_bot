#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tradebot as tb
import numpy as np
import copy
import progressbar
import pickle
import klepto


def prepare_bbox():
    global n_features, n_actions, max_time

    # Reset environment to the initial state, just in case
    if bbox.is_level_loaded():
        bbox.reset_level()


def calc_best_action_using_checkpoint(action_range=4):
    # Pretty straightforward — we create a checkpoint and get it's ID
    checkpoint_id = bbox.create_checkpoint()

    best_action = -1
    best_score = -1e9

    for action in range(n_actions):
        for _ in range(action_range):  # random.randint(1,100)
            bbox.do_action(action)

        if bbox.get_score() > best_score:
            best_score = bbox.get_score()
            best_action = action

        bbox.load_from_checkpoint(checkpoint_id)

    bbox.clear_all_checkpoints()

    return best_action


def get_best_action(action_range=4):
    # Pretty straightforward — we create a checkpoint and get it's ID
    checkpoint_id = bbox.create_checkpoint()

    best_action = -1
    best_score = -1e9

    for action in range(n_actions):
        for _ in range(action_range):  # random.randint(1,100)
            bbox.do_action(action)

        if bbox.get_score() > best_score:
            best_score = bbox.get_score()
            best_action = action

        bbox.load_from_checkpoint(checkpoint_id)

    bbox.clear_all_checkpoints()

    return best_action


def train_minibatch(minibatch):
    old_state_s = np.array([row[0] for row in minibatch])
    action_s = np.array([row[1] for row in minibatch])
    reward_s = np.array([row[2] for row in minibatch])
    new_state_s = np.array([row[3] for row in minibatch])
    old_qwal_s = model.predict(old_state_s, batch_size=32)
    newQ_s = model.predict(new_state_s, batch_size=32)
    maxQ_s = np.max(newQ_s, axis=1)
    y = old_qwal_s
    update_s = reward_s + gamma * maxQ_s
    for i in range(len(action_s)):
        y[i, action_s[i]] = update_s[i]

    model_prim.fit(old_state_s, y, batch_size=batchSize, nb_epoch=1, verbose=0)
    return



def run_bbox(verbose=False, epsilon=0.1, gamma=0.99, action_repeat=5, update_frequency=4, sample_fit_size=32,
             replay_memory_size=100000,
             load_weights=False, save_weights=False):
    global pgi
    has_next = 1
    global actions
    global bbox
    # Prepare environment - load the game level
    prepare_bbox()

    update_frequency_cntr = 0

    h = 0
    if load_weights:
        model.load_weights(root + 'my_model_weights.h5')
        model_prim.load_weights(root + 'my_model_weights.h5')
    # stores tuples of (S, A, R, S')


    while has_next:
        # Get current environment state

        pgi += 1
        if pgi % print_step == 0:
            bar.update(pgi)

       # state = copy.copy(bbox.get_state())
        state = bbox.get_state()
        train_states_logs.append((state.flatten().tolist())[0:-4])

        prev_reward = copy.copy(bbox.get_score())

        # Run the Q function on S to get predicted reward values on all the possible actions
        qval = model.predict(state.reshape(1, n_features), batch_size=1)
        train_qval.append(qval)

        action = (np.argmax(qval))
        actions[action] += 1
        # Choose an action to perform at current step
        if random.random() < epsilon:  # choose random action or best action
            action = np.random.randint(0, n_actions)  # assumes 4 different actions
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))

        # Perform chosen action, observe new state S'
        # Function do_action(action) returns False if level is finished, otherwise returns True.


        for a in range(action_repeat):
            has_next = bbox.do_action(action)
        new_state = copy.copy(bbox.get_state())
        reward = copy.copy(bbox.get_score()) - prev_reward

        #if random.random() < 0.2 or reward > 0 :  # в запоминаем все успешные действия и только 20% нейспешных
        if True:  # в запоминаем все успешные действия и только 20% нейспешных
            if (len(replay) < replay_memory_size):  # if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
            else:  # if buffer full, overwrite old values
                h=np.random.randint(0,replay_memory_size-1)
                replay[h] = (state, action, reward, new_state)

                # randomly sample our experience replay memory
                # minibatch = random.sample(replay, batchSize)
                minibatch = random.sample(replay, sample_fit_size)
                train_minibatch(minibatch=minibatch)

                if update_frequency_cntr >= update_frequency:
                    prim_weights = model_prim.get_weights()

                    model.set_weights(prim_weights)
                    update_frequency_cntr = 0
                update_frequency_cntr += 1
                # step_times.append(time.time()-st)

    # Finish the game simulation, print earned reward and save weights
    if save_weights:
        model_prim.save_weights(root + 'my_model_weights.h5', overwrite=True)
    bbox.finish(verbose=0)


from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.regularizers import l1l2,activity_l1l2

import random

random.seed(6)
n_features = n_actions = max_time = -1
days_to_train =-1
first_run =False
resumple=False

replay_file = u'e:\\trade_data\\HistoryData\\replay.klp'
#bot_file_name = u'e:\\trade_data\\HistoryData\\train_50x40_data_2016.bot'
#u'e:\\trade_data\\HistoryData\\train_50x40_data_2015-2016.bot
bot_file_name = u'e:\\trade_data\\HistoryData\\Ri_train_50x40_data_2015-2016.bot'
d = klepto.archives.dir_archive(bot_file_name, cached=True, serialized=True)
d.load("bbox")
bbox = d["bbox"]
del d
if days_to_train != -1:
    bbox.set_sample_days(days_to_train)

exploration_epochs = 1
learning_epochs =1

gamma = 0.8  # a high gamma makes a long term reward more valuable
epsilon=0.1
action_repeat = 3 # repeat each action this many times // было 4
update_frequency = 50  # the number of time steps between each Q-net update
batchSize = 32  # параметр для обучения сети
l1_reg=0.05
l2_reg=0.00001
#replay_memory_size = np.minimum(int(bbox.total_steps / float(action_repeat)), 500000 ) # размер памяти, буфера
replay_memory_size=200000
print('replay_memory_size ', replay_memory_size)
sample_fit_size = 128  # Размер минибатча, по которому будет делаться выборка из буфера
print_step = 10

n_features = bbox.get_num_of_features()  # учесть что мы сдесь получаем шайп
print('n_features=', n_features)
n_actions = bbox.get_num_of_actions()
max_time = bbox.get_max_time()



model = Sequential()
model.add(Dense(n_features, init='lecun_uniform', input_shape=(n_features,)))
model.add(Activation('relu'))

model.add(Dense(1600, init='lecun_uniform',
                W_regularizer=l1l2(l1=l1_reg,l2=l2_reg)
                 ))  # a 10 neuron network gives better than random result
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(800, init='lecun_uniform',
          W_regularizer=l1l2(l1=l1_reg, l2=l2_reg)
           ))  # a 10 neuron network gives better than random result
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(n_actions, init='lecun_uniform'))
model.add(Activation('linear'))  # linear output so we can have range of real-valued outputs
rms = RMSprop(lr=0.00025)  # 0.00025
model.compile(loss='mse', optimizer=rms)

json_string = model.to_json()

root = u'e:\\trade_data\\HistoryData\\'

open(root + 'my_model_architecture.json', 'w').write(json_string)
model_prim = model_from_json(open(root + 'my_model_architecture.json').read())

model_prim.compile(loss='mse', optimizer=rms)

r = klepto.archives.dir_archive(replay_file, cached=True, serialized=True)
if not first_run:  # "загружаем веса, если запуск не первый"
    model.load_weights(root + 'my_model_weights.h5')
    model_prim.load_weights(root + 'my_model_weights.h5')

    r.load("replay")
    replay = r['replay']
else:
    replay = []
    r['replay'] = replay

load_weights = False

replay = []
#r['replay'] = replay

pgi = 0
total_steps = int((exploration_epochs + learning_epochs) * bbox.total_steps / float(action_repeat))
bar = progressbar.ProgressBar(maxval=total_steps)
bar.start()

#текстовые логи
train_states_logs=[]
train_qval=[]
test_states_logs=[]
test_qval=[]


for i in range(exploration_epochs):
    print("exploration ", i, " of ", exploration_epochs)
    epsilon_t=1.0
    actions = np.array([0, 0, 0])
    run_bbox(verbose=0, epsilon=epsilon, gamma=gamma, action_repeat=action_repeat,
             update_frequency=update_frequency, sample_fit_size=sample_fit_size,
             replay_memory_size=replay_memory_size, load_weights=load_weights, save_weights=True)
    print("score: ", np.round(bbox.get_score()), actions)

    if epsilon_t > 0.1:
        epsilon_t -= (1.0 / exploration_epochs)  # потихоньку увеличиваем вероятность использования знаний
    if resumple:
        bbox.set_sample_days(days_to_train)

    total_steps = int((exploration_epochs + learning_epochs) * bbox.total_steps / float(action_repeat))
    bar = progressbar.ProgressBar(maxval=total_steps)
r.dump()

for i in range(learning_epochs):
    actions = np.array([0, 0, 0])
    print("learning ", i, " of ", learning_epochs)
    epsilon = 0.1
    run_bbox(verbose=0, epsilon=epsilon, gamma=gamma, action_repeat=action_repeat,
             update_frequency=update_frequency, sample_fit_size=sample_fit_size,
             replay_memory_size=replay_memory_size, load_weights=load_weights, save_weights=True)
    print("score: ", np.round(bbox.get_score()), actions)

    if resumple:
        bbox.set_sample_days(days_to_train)

    total_steps = int((exploration_epochs + learning_epochs) * bbox.total_steps / float(action_repeat))
    bar = progressbar.ProgressBar(maxval=total_steps + 100)
r.dump()



def test_strategy(n=4, resample=True, action_repeat=6):
    results = []

    for i in range(n):

        if resample:
            random.seed(1 + i)
            bbox.set_sample_days(days_to_train)

        bbox.reset_level()
        has_next = True
        actions = np.array([0, 0, 0])
        while has_next:
            state = bbox.get_state()
            qval = model.predict(state.reshape(1, n_features), batch_size=1)


            action = (np.argmax(qval))
            actions[action] += 1
            for a in range(action_repeat):
                has_next = bbox.do_action(action)

        bbox.finish(verbose=0)
        print(" test ", i, " score: ", bbox.get_score(), actions)

    return results

print ('тест на тренировочных данных')
test_times = 1
results = test_strategy(test_times, action_repeat=action_repeat, resample=False)
with open('test_states.txt', "w") as file:
    for row in test_states_logs:
        file.write(str(list(row)) + '\n')
    file.flush()
    file.close()

print (train_states_logs==test_states_logs)