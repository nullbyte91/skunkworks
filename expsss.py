#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 04:50:08 2020

@author: naraharib
"""
import os
#os.chdir("/home/naraharib/Narahari/Personal/future/skunkworks/DDPG-her")
import gym
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from play import Play
import mujoco_py
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch



ENV_NAME = "FetchPickAndPlace-v1"

test_env = gym.make(ENV_NAME)

obs_dict = test_env.reset()

test_env.render()

a = test_env.action_space.low
obs_dash = test_env.step(a)
test_env.render()
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.DataFrame({"start":obs_dict["observation"],"1st":obs_dash[0]["observation"]})

df["diff1"] = df["1st"] - df["start"]
df



a = np.array([1, 1, 1, 1])
obs_dash = test_env.step(a)
test_env.render()

df["2"] = pd.Series(obs_dash[0]["observation"])

df["diff2-1"] = df["2"] - df["1st"]
df["diff2-start"] = df["2"] - df["start"]
df

import time


#actions
#front/back, left/right, top/bottom, release/pick
b = 1
for i in range(1):
    b *= -1
    a = np.array([0, 0, -1, -1])
    obs_dash = test_env.step(a)
    test_env.render()
    #time.sleep(2)
obs_dash
df["3"] = pd.Series(obs_dash[0]["observation"])
df["4"] = pd.Series(obs_dash[0]["observation"])
df["diff4-3"] = (df["4"] - df["3"]) == 0
df

obs_dash[0]["achieved_goal"]
