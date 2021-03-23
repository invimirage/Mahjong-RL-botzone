#!/usr/bin/env python
# encoding: utf-8
'''
@file: filereader.py
@time: 2020/12/19 18:34
@desc:
'''
# _*_coding:utf-8_*_
import time, threading
import os
import json
import argparse

parser = argparse.ArgumentParser(description='Model')
parser.add_argument('-lp', '--load_path', type=str, default=r'C:\Users\zrf19\Desktop\大四上\强化学习\麻将\mjdata\output2017',
                    help='from where to load raw data')
parser.add_argument('-sp', '--save_path', type=str, default='training_data', help='where to save parsed training data')
parser.add_argument('-tn', '--thread_number', type=int, default=32,
                    help='thread number in total, for thread i, it parses files i + k * thread_number, k = 0, 1, 2...')
parser.add_argument('-tr', '--thread_round', type=int, default=4,
                    help='how many rounds the threads are run, tn // tr = threads running simultaneously')
args = parser.parse_args()

def round(outcome, fanxing, score, fname, zhuangjia, requests, responses):
    return {
    'outcome' : outcome,
    'fanxing' : fanxing,
    'score' : score,
    'fname' : fname,
    'zhuangjia': zhuangjia,
    'requests' : requests,
    'responses' : responses
    }

class Reader(threading.Thread):
    def __init__(self, files, id):
        super(Reader, self).__init__()
        self.files = files
        self.id = id

    def run(self):
        self.rounds = []
        self.removed_hua = 0
        self.removed_cuohu = 0
        self.removed_fan = 0
        for filenum, file in enumerate(self.files):
            if filenum % 1000 == 0:
                print('{}/{}'.format(filenum, len(self.files)))
            with open(file, encoding='utf-8') as f:
                line = f.readline()
                count = 1
                requests = [[], [], [], []]
                responses = [[], [], [], []]
                zhuangjia = 0
                outcome = ''
                fanxing = []
                score = 0
                fname = file
                flag = True
                already_hu = False
                draw_hua = False
                while line:
                    # print(line)
                    line = line.strip('\n').split('\t')
                    if count == 2:
                        # print(line)
                        outcome = line[-1]
                        fanxing = list(map(lambda x: x.strip("'"), line[2].strip('[]').split(',')))
                        if outcome != '荒庄':
                            fanshu = 0
                            for fan in fanxing:
                                try:
                                    description, this_score = fan.split('-')
                                except:
                                    description = fan
                                    if description == '全带五':
                                        this_score = 16
                                    elif description == '三同刻':
                                        this_score = 16
                                    else:
                                        print(fan)
                                if description == '花牌':
                                    continue
                                fanshu += int(this_score)
                            if fanshu < 8:
                                flag = False
                                self.removed_fan += 1
                                break
                        score = int(line[1])
                        # quan = line[0]
                        # print(fanxing)
                    if count > 2 and count <= 6:
                        playerID = int(line[0])
                        request = "1 0 0 0 0 "
                        cards = line[1]
                        cards = list(map(lambda x: x.strip("'"), cards.strip('[]').split(',')))
                        if len(cards) == 14:
                            draw_card = cards[-1]
                            hua_count = 0
                            for card in cards:
                                if 'H' in card:
                                    hua_count += 1
                                    draw_card = card
                                    draw_hua = True
                            if hua_count > 1:
                                self.removed_hua += 1
                                flag = False
                                break
                            cards.remove(draw_card)
                            zhuangjia = playerID
                        else:
                            if 'H' in ' '.join(cards):
                                self.removed_hua += 1
                                flag = False
                                break
                            draw_card = None
                        request += (' '.join(cards))
                        requests[playerID].append(request)
                        if draw_card is not None:
                           requests[playerID].append('2 ' + draw_card)
                           responses[playerID].append("PASS")
                        else:
                            requests[playerID].append('3 {} DRAW'.format(zhuangjia))
                            responses[playerID].append("PASS")
                    if count > 6:
                        playerID = int(line[0])
                        action = line[1]
                        cards = line[2]
                        if draw_hua and action != '补花':
                            self.removed_hua += 1
                            flag = False
                            break
                        if action == '吃':
                            middel_card = list(map(lambda x: x.strip("'"), cards.strip('[]').split(',')))[1]
                            next_line = f.readline()
                            play_card = next_line.strip('\n').split('\t')[2]
                            play_card = play_card.strip("[]'")
                            _cards = [middel_card, play_card]
                            _action = action
                        elif action == '碰':
                            next_line = f.readline()
                            play_card = next_line.strip('\n').split('\t')[2]
                            play_card = play_card.strip("[]'")
                            _cards = [play_card]
                            _action = action
                        elif action == '补花':
                            # flag = False
                            draw_hua = False
                            for i in range(4):
                                requests[i].pop()
                                responses[i].pop()
                            # if not flag:
                            #     self.removed_hua += 1
                            #     break
                            line = f.readline()
                            if not line:
                                for i in range(4):
                                    requests[i].pop()
                                last_hua = False
                                for i in range(4):
                                    if 'H' in requests[i][-1]:
                                        last_hua = True
                                if last_hua:
                                    for i in range(4):
                                        requests[i].pop()
                                        responses[i].pop()
                            count += 1
                            continue
                        else:
                            card = list(map(lambda x: x.strip("'"), cards.strip('[]').split(',')))[0]
                            _cards = [card]
                            _action = action
                            if action == '和牌':
                                already_hu = True
                            if action == '摸牌' or action == '补花后摸牌' or action == '杠后摸牌':
                                if 'H' in card:
                                    draw_hua = True
                        for i in range(4):
                            request = get_request(_action, playerID, _cards, i)
                            # print(request)
                            response = get_response(_action, playerID, _cards, i)
                            # print(response)

                            requests[i].append(request)
                            if response is not None:
                                responses[i].append(response)

                    line = f.readline()
                    # 胡牌之后就没有了
                    if line and already_hu:
                        self.removed_cuohu += 1
                        flag = False
                        # print(fname)
                        break
                    if not line:
                        for i in range(4):
                            requests[i].pop()
                        last_hua = False
                        for i in range(4):
                            if 'H' in requests[i][-1]:
                                last_hua = True
                        if last_hua:
                            for i in range(4):
                                requests[i].pop()
                                responses[i].pop()
                    count += 1
                if flag:
                    self.rounds.append(round(outcome, fanxing, score, fname, zhuangjia, requests, responses))

    def get_res(self):
        with open('{}/Tread {}-mini.json'.format(args.save_path, self.id), 'w') as file_obj:
            json.dump(self.rounds, file_obj)
        return self.removed_hua, self.removed_cuohu, self.removed_fan

def get_request(action, playerid, cards, myplayerid):
    playerid = str(playerid)
    myplayerid = str(myplayerid)
    request = None
    if action == '打牌':
        request = ['3', playerid, 'PLAY', cards[0]]
    if action == '摸牌' or action == '补花后摸牌' or action == '杠后摸牌':
        if playerid == myplayerid:
            request = ['2', cards[0]]
        else:
            request = ['3', playerid, 'DRAW']
    if action == '吃':
        request = ['3', playerid, 'CHI'] + cards
    if action == '碰':
        request = ['3', playerid, 'PENG', cards[0]]
    if action == '明杠' or action == '暗杠':
        request = ['3', playerid, 'GANG']
    if action == '补杠':
        request = ['3', playerid, 'BUGANG', cards[0]]
    if request is None:
        return None
    return ' '.join(request)

def get_response(action, playerid, cards, myplayerid):
    if playerid != myplayerid:
        response = ['PASS']
    else:
        if action == '打牌':
            response = ['PLAY', cards[0]]
        if action == '摸牌' or action == '补花后摸牌' or action == '杠后摸牌':
           response = ['PASS']
        if action == '吃':
            response = ['CHI'] + cards
        if action == '碰':
            response = ['PENG', cards[0]]
        if action == '明杠':
            response = ['GANG']
        if action == '暗杠':
            response = ['GANG', cards[0]]
        if action == '补杠':
            response = ['BUGANG', cards[0]]
        if action == '和牌':
            response = ['HU']
    # print(action)
    return ' '.join(response)


if __name__ == '__main__':
    # reader = Reader(['C:\\Users\\zrf19\\Desktop\\强化学习\\麻将\\mjdata\\output2017/PLAY/2017-07-29-305.txt'], 10086)
    # reader.start()
    # reader.join()
    # reader.get_res()
    #线程数量
    thread_num = args.thread_number
    thread_rounds = args.thread_round
    thread_per_round = thread_num // thread_rounds
    #起始时间
    t = []
    folder = args.load_path
    dirs = os.listdir(folder)
    files = []
    for dir in dirs:
        subfolder = folder + '/' + dir
        for file in os.listdir(subfolder):
            files.append(subfolder + '/' + file)
    # files = files[:10000]
    filenum = len(files)
    #生成线程
    for i in range(thread_num):
        t.append(Reader(files[i::thread_num], i))
    rm_hua = 0
    rm_cuohu = 0
    rm_fan = 0
    for this_round in range(thread_rounds):
        #开启线程
        for i in range(thread_per_round):
            t[i+thread_per_round*this_round].start()
        for i in range(thread_per_round):
            t[i+thread_per_round*this_round].join()
            r_h, r_c, r_f = t[i+thread_per_round*this_round].get_res()
            rm_hua += r_h
            rm_cuohu += r_c
            rm_fan += r_f
    print("一共{}局记录，其中补花错误{}局，错和{}局，番数不足{}局，训练数据共{}局".format(filenum, rm_hua, rm_cuohu, rm_fan,
                                                            filenum - rm_hua - rm_cuohu - rm_fan))