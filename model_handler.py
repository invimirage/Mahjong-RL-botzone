#!/usr/bin/env python
# encoding: utf-8
'''
@file: filereader.py
@time: ？？？？
缩小模型供botzone使用
'''
import torch
import argparse

parser = argparse.ArgumentParser(description='Building the smaller model')
parser.add_argument('-o', '--old_path', type=str, default='../models/super_model_2', help='path to original model')
parser.add_argument('-n', '--new_path', type=str, default='../models/super_model_small', help='path to smaller model')

args = parser.parse_args()
checkpoint = torch.load(args.old_path)
state = {'model': checkpoint['model']}
torch.save(state, args.new_path, _use_new_zipfile_serialization=False)