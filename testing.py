from DQN import DQN
import gym
import sys
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.misc import imresize

if __name__ == "main":
  # hyperparameters etc
  gamma = 0.99
  batch_sz = 32
  num_episodes = 500
  total_t = 0
  experience_replay_buffer = []
  episode_rewards = np.zeros(num_episodes)
  last_100_avgs = []

  # epsilon for Epsilon Greedy Algorithm
  epsilon = 1.0
  epsilon_min = 0.1
  epsilon_change = (epsilon - epsilon_min) / 500000

  env = gym.envs.make("Breakout")