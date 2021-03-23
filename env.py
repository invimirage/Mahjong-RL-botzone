#!/usr/bin/env python
# encoding: utf-8
'''
@author: zrf
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: deamoncao100@gmail.com
@software:XXXX
@file: DL_agent.py
@time: 2020/12/10 13:27
@desc:
'''
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append("..")
import json
from MahjongGB import MahjongFanCalculator
import torch
import torch.optim as optim
from enum import Enum
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import math
from copy import deepcopy
from torch.distributions import Categorical
import torch.multiprocessing as mp
import argparse
import time
from DL_agent import agent as dl_agent
# from botzone.new_main import agent as dl_search_agent0
# from botzone.new_main_1 import agent as dl_search_agent1
# from new_main_2 import agent as dl_search_agent2
# from new_main_3 import agent as dl_search_agent3

parser = argparse.ArgumentParser(description='the environment to test bots')
parser.add_argument('-o', '--old_path', type=str, default='../models/super_model_2', help='path to stable model')
parser.add_argument('-n', '--new_path', type=str, default='../models/rl_pg_new', help='path to trained model')
parser.add_argument('-p', '--num_process_per_gpu', type=int, default=1, help='number of processes to run per gpu')
parser.add_argument('-pi', '--print_interval', type=int, default=500, help='how often to print')

args = parser.parse_args()

class requests(Enum):
    initialHand = 1
    drawCard = 2
    DRAW = 4
    PLAY = 5
    PENG = 6
    CHI = 7
    GANG = 8
    BUGANG = 9
    MINGGANG = 10
    ANGANG = 11

class responses(Enum):
    PASS = 0
    PLAY = 1
    HU = 2
    # 需要区分明杠和暗杠
    MINGGANG = 3
    ANGANG = 4
    BUGANG = 5
    PENG = 6
    CHI = 7
    need_cards = [0, 1, 0, 0, 1, 1, 0, 1]
    loss_weight = [1, 1, 5, 2, 2, 2, 2, 2]

class cards(Enum):
    # 饼万条
    B = 0
    W = 9
    T = 18
    # 风
    F = 27
    # 箭牌
    J = 31


class ActorCritic(nn.Module):
    def __init__(self, card_feat_depth, num_extra_feats, num_cards, num_actions):
        super().__init__()
        hidden_channels = [8, 16, 32]
        hidden_layers_size = [512, 1024]
        linear_length = hidden_channels[1] * num_cards * card_feat_depth
        self.linear_length = linear_length + num_extra_feats
        # self.number_card_net = nn.Sequential(
        #     nn.Conv2d(3, hidden_channels[0], 3, stride=1, padding=1),
        #     nn.ReLU(),
        # )
        self.card_net = nn.Sequential(
            nn.Conv2d(1, hidden_channels[0], 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[0], hidden_channels[1], 5, stride=1, padding=2),
        )
        self.card_play_decision_net = nn.Sequential(
            nn.Linear(num_extra_feats + linear_length, hidden_layers_size[0]),
            nn.Sigmoid(),
            nn.Linear(hidden_layers_size[0], hidden_layers_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[1], num_cards),
            nn.Softmax(dim=1)
        )
        self.chi_peng_decision_net = nn.Sequential(
            nn.Linear(num_extra_feats + linear_length, hidden_layers_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[0], num_cards),
            nn.Softmax(dim=1)
        )
        self.action_decision_net = nn.Sequential(
            nn.Linear(num_extra_feats + linear_length, hidden_layers_size[1]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[1], num_actions),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(num_extra_feats + linear_length, hidden_layers_size[0]),
            nn.ReLU(),
            nn.Linear(hidden_layers_size[0], hidden_layers_size[1]),
            nn.Sigmoid(),
            nn.Linear(hidden_layers_size[1], 2048),
            nn.ReLU(),
            nn.Linear(2048, 1)
        )

    # play, chi_gang,
    def forward(self, card_feats, extra_feats, device, decide_which, mask):
        assert decide_which in ['play', 'chi_gang', 'action']
        card_feats = torch.from_numpy(card_feats).to(device).unsqueeze(1).to(torch.float32)
        card_layer = self.card_net(card_feats)
        batch_size = card_layer.shape[0]
        extra_feats_tensor = torch.from_numpy(extra_feats).to(torch.float32).to(device)
        linear_layer = torch.cat((card_layer.view(batch_size, -1), extra_feats_tensor), dim=1)
        mask_tensor = torch.from_numpy(mask).to(torch.float32).to(device)
        if decide_which == 'play':
            card_probs = self.card_play_decision_net(linear_layer)
            valid_card_play = self.mask_unavailable_actions(card_probs, mask_tensor)
            return valid_card_play
        elif decide_which == 'action':
            # print(linear_layer.shape)
            action_probs = self.action_decision_net(linear_layer)
            valid_actions = self.mask_unavailable_actions(action_probs, mask_tensor)
            # print(valid_actions, valid_card_play)
            return valid_actions
        else:
            card_probs = self.chi_peng_decision_net(linear_layer)
            valid_card_play = self.mask_unavailable_actions(card_probs, mask_tensor)
            return valid_card_play

    def mask_unavailable_actions(self, result, valid_actions_tensor):
        valid_actions = result * valid_actions_tensor
        if valid_actions.sum() > 0:
            masked_actions = valid_actions / valid_actions.sum()
        else:
            masked_actions = valid_actions_tensor / valid_actions_tensor.sum()
        return masked_actions


class MahjongEnv:
    def __init__(self):
        self.test = True
        self.total_cards = 34
        self.total_actions = len(responses) - 2
        self.print_interval = args.print_interval
        self.round_count = 0
        # state = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
        # torch.save(state, self.model_path, _use_new_zipfile_serialization=False)
        self.bots = []
        self.winner = np.zeros(4)
        self.dianpaoer = np.zeros(4)
        self.winning_rate = np.zeros(4)
        self.win_steps = []
        self.losses = []
        self.scores = np.zeros(4)
        # 0, 2为老model
        if args.old_path == args.new_path:
            model_path = args.old_path
            self.bots = [dl_agent(model_path), dl_search_agent0(model_path), dl_agent(model_path), dl_search_agent1(model_path)]
            # for i in range(4):
            #     if i != 0:
            #         self.bots.append(dl_agent(model_path))
            #     else:
            #         self.bots.append(dl_search_agent0(model_path))
        else:
            for i in range(4):
                if i % 2 == 0:
                    self.bots.append(dl_agent(args.old_path))
                else:
                    self.bots.append(dl_agent(args.new_path))
        self.reset(True)


    def reset_for_test(self, initial=False, global_counter=0):
        self.round_count += 1
        if self.round_count % 4 == 0:
            self.reset(initial, global_counter)
        else:
            self.tile_wall = deepcopy(self.doc_tile)
            self.men = (self.men + 1) % 4
            self.bots_order = [self.bots[(i + self.men) % 4] for i in range(4)]
            self.turnID = 0
            self.drawer = 0
            for bot, reward in zip(self.bots, self.scores):
                bot.reset()


    def reset(self, initial=False, global_counter=0, global_winning_rate=None):
        all_tiles = np.arange(self.total_cards)
        all_tiles = all_tiles.repeat(4)
        np.random.shuffle(all_tiles)
        # 用pop，从后面摸牌
        self.tile_wall = np.reshape(all_tiles, (4, -1)).tolist()
        self.doc_tile = deepcopy(self.tile_wall)
        self.quan = np.random.choice(4)
        self.men = np.random.choice(4)

        # 这一局bots的order，牌墙永远下标和bot一致
        self.bots_order = [self.bots[(i + self.men) % 4] for i in range(4)]
        self.turnID = 0
        self.drawer = 0
        if not initial:
            for bot, reward in zip(self.bots_order, self.scores):
                bot.reset()
            if self.round_count % (4 * self.print_interval) == 0:
                win_sum = self.winner.sum()
                total_rounts = 4 * self.print_interval
                print(
                    '目前进行了{}轮，在前{}轮中，new bot winning rate: {:.2%}, new bot total score: {},'
                    'new bot dianpao: {}, old bot score: {}, dianpao: {},  old bot winning rate: {:.2%}\n'
                    ' 和牌{}局，荒庄{}局，和牌率{:.2%}，平均和牌回合数{}'.format(
                        self.round_count, total_rounts, self.winner[1::2].sum() / win_sum, self.scores[1::2].sum(),
                        self.dianpaoer[1::2].sum(), self.scores[::2].sum(), self.dianpaoer[::2].sum(),
                        self.winner[::2].sum() / win_sum, self.winner.sum(),
                                                               total_rounts - self.winner.sum(),
                                                               self.winner.sum() / total_rounts,
                                                               sum(self.win_steps) / self.winner.sum()
                    ))
                self.winner = np.zeros(4)
                self.dianpaoer = np.zeros(4)
                self.scores = np.zeros(4)
                self.win_steps = []


    def run_round(self):
        fan_count = 0
        player_id = 0
        dianpaoer = None
        outcome = ''
        player_responses = []
        self.drawn = False
        while True:
            if self.turnID == 0:
                for id, player in enumerate(self.bots_order):
                    player_responses.append(player.step('0 %d %d' % (id, self.quan)))
            elif self.turnID == 1:
                player_responses = []
                for id, player in enumerate(self.bots_order):
                    request = ['1']
                    for i in range(4):
                        request.append('0')
                    for i in range(13):
                        request.append(self.getCardName(self.tile_wall[id].pop()))
                    request = ' '.join(request)
                    player_responses.append(player.step(request))
            else:
                requests = self.parse_response(player_responses)
                if requests[0] in ['hu', 'huangzhuang']:
                    outcome = requests[0]
                    if outcome == 'hu':
                        player_id = int(requests[1])
                        fan_count = int(requests[2])
                        dianpaoer_id = requests[3]
                        self.winner[self.bots.index(self.bots_order[player_id])] += 1
                        self.scores[self.bots.index(self.bots_order[player_id])] += fan_count
                        if dianpaoer_id != 'None':
                            dianpaoer_id = int(dianpaoer_id)
                            self.scores[self.bots.index(self.bots_order[dianpaoer_id])] -= 0.5 * fan_count
                            self.dianpaoer[self.bots.index(self.bots_order[dianpaoer_id])] += 1
                        self.win_steps.append(self.turnID)
                    break
                else:
                    player_responses = []
                    for i in range(4):
                        player_responses.append(self.bots_order[i].step(requests[i]))
            self.turnID += 1
        # print('{} {}'.format(outcome, fan_count))
        # difen = 8
        # if outcome == 'hu':
        #     for i in range(4):
        #         if i == player_id:
        #             self.scores[i] = 20
        #         elif i == dianpaoer:
        #             self.scores[i] = -5
        #         else:
        #             self.scores[i] = -1

            # for i in range(4):
            #     if i == player_id:
            #         self.scores[i] = 10
                #     if dianpaoer is None:
                #         # 自摸
                #         self.scores[i] = 3 * (difen + fan_count)
                #     else:
                #         self.scores[i] = 3 * difen + fan_count
                # else:
                #     if dianpaoer is None:
                #         self.scores[i] = -0.5 * (difen + fan_count)
                #     else:
                #         if i == dianpaoer:
                #             self.scores[i] = -2 * (difen + fan_count)
                #         else:
                #             self.scores[i] = -0.5 * difen
        # print(self.scores)

    def parse_response(self, player_responses):
        requests = []
        for id, response in enumerate(player_responses):
            response = response.split(' ')
            response_name = response[0]
            if response_name == 'HU':
                return ['hu', id, response[1], response[2]]
            if response_name == 'PENG':
                requests = []
                for i in range(4):
                    requests.append('3 %d PENG %s' % (id, response[1]))
                self.drawer = (id + 1) % 4
                break
            if response_name == "GANG":
                requests = []
                for i in range(4):
                    requests.append('3 %d GANG' % (id))
                self.drawer = id
                break
            if response_name == 'CHI':
                for i in range(4):
                    requests.append('3 %d CHI %s %s' % (id, response[1], response[2]))
                self.drawer = (id + 1) % 4
            if response_name == 'PLAY':
                for i in range(4):
                    requests.append('3 %d PLAY %s' % (id, response[1]))
                self.drawer = (id + 1) % 4
            if response_name == 'BUGANG':
                for i in range(4):
                    requests.append('3 %d BUGANG %s' % (id, response[1]))
                self.drawer = id
        # 所有人pass，摸牌
        if len(requests) == 0:
            if len(self.tile_wall[self.drawer]) == 0:
                return ['huangzhuang', 0]
            draw_card = self.tile_wall[self.drawer].pop()
            for i in range(4):
                if i == self.drawer:
                    requests.append('2 %s' % self.getCardName(draw_card))
                else:
                    requests.append('3 %d DRAW' % self.drawer)
        return requests

    def getCardInd(self, cardName):
        return cards[cardName[0]].value + int(cardName[1]) - 1

    def getCardName(self, cardInd):
        num = 1
        while True:
            if cardInd in cards._value2member_map_:
                break
            num += 1
            cardInd -= 1
        return cards(cardInd).name + str(num)

import time
def train_thread(global_episode_counter):

    env = MahjongEnv()
    while True:
        env.run_round()
        env.reset_for_test(False, global_episode_counter.value)
        global_episode_counter.value += 1


def main():
    num_processes_per_gpu = args.num_process_per_gpu
    new_model_path = args.new_path
    mp.set_start_method('spawn')  # required to avoid Conv2d froze issue
    # critic
    gpu_count = torch.cuda.device_count()
    num_processes = gpu_count * num_processes_per_gpu

    # multiprocesses, Hogwild! style update
    processes = []
    init_episode_counter_val = 0
    global_episode_counter = mp.Value('i', init_episode_counter_val)
    # each worker_thread creates its own environment and trains agents
    for rank in range(num_processes):
        # only write summaries in one of the workers, since they are identical
        # worker_summary_queue = summary_queue if rank == 0 else None
        worker_thread = mp.Process(
            target=train_thread, args=(global_episode_counter, ))
        worker_thread.daemon = True
        worker_thread.start()
        processes.append(worker_thread)
        time.sleep(2)

    # wait for all processes to finish
    try:
        killed_process_count = 0
        for process in processes:
            process.join()
            killed_process_count += 1 if process.exitcode == 1 else 0
            if killed_process_count >= num_processes:
                # exit if only monitor and writer alive
                raise SystemExit
    except (KeyboardInterrupt, SystemExit):
        for process in processes:
            # without killing child process, process.terminate() will cause orphans
            # ref: https://thebearsenal.blogspot.com/2018/01/creation-of-orphan-process-in-linux.html
            # kill_child_processes(process.pid)
            process.terminate()
            process.join()

if __name__ == '__main__':
    main()