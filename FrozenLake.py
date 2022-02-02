# **
# * Author:    Chris Ryan
# * Created:   11.09.2021
# ** 
# Q Learning approach to FrozenLake-v0 from https://gym.openai.com (playground for reinforcement learning AI).
# actions: up, down, right, left
# SFFF  (S: starting point, safe)
# FHFH  (F: frozen surface, safe but it is possible to slip)
# FFFH  (H: hole, fall to your doom)
# HFFG  (G: goal)

# Hyperparameters (Bellman equation)
# α (alpha)   The learning rate (0.001). Dictates how much existing best known values are impacted by
# γ (gamma)   The reward discount factor. Normally between 0.90-0.99). Favours shorter-term rewards
# ε (epsilon) Governs the balance between exploratory (random) action and greedy (best known) action.
#             This needs to decrease over time to favour best known action.

import gym
import random
import numpy as np
from collections import deque
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

EPISODES = 1000
GAMMA = 0.95            # γ (gamma) The reward discount factor. Normally between 0.90-0.99). Favours shorter-term rewards
LEARNING_RATE = 0.005   # α (alpha) The learning rate (0.001). Dictates how much existing best known values are impacted by
MEMORY_SIZE = 1000000
BATCH_SIZE = 20
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNSolver:
    def __init__(self):
        self.observation_space = env.observation_space.n 
        self.action_space = env.action_space.n
        self.epsilon = EXPLORATION_MAX
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = keras.Sequential()
        self.model.add(layers.Dense(24, input_shape=(env.observation_space.n,), activation="relu"))
        self.model.add(layers.Dense(env.action_space.n, activation="linear"))
        self.model.compile(loss="mse", optimizer=optimizers.Adam(lr=LEARNING_RATE))

    def epsilon_decay(self):
        if self.epsilon > EXPLORATION_MIN:
            self.epsilon *= EXPLORATION_DECAY

    def remember(self, state, action, reward, new_state, done):
        self.memory.append((state, action, reward, new_state, done))

    def act(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, env.action_space.n)
        else:
            action = np.argmax(self.model.predict(np.identity(env.observation_space.n)[state:state + 1]))
        return action

    def q_update(self, state, action, reward, new_state, done):
        if not done:
            target = reward + GAMMA * np.max(self.model.predict(np.identity(self.observation_space)[new_state:new_state + 1]))
        else:
            target = reward
        target_vector = self.model.predict(np.identity(self.observation_space)[state:state + 1])[0]
        target_vector[action] = target
        self.model.fit(np.identity(self.observation_space)[state:state + 1], target_vector.reshape(-1, self.action_space), epochs=1, verbose=0)

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, new_state, done in batch:
            self.q_update(state, action, reward, new_state, done)    
    
    def report(self, episode, steps, reward):
        colour = '\033[92m' if reward > 0 else '\033[91m'
        print("Episode: " + str(episode).rjust(4) + '  ε: {:.3f}'.format(self.epsilon) + "  Steps: " + str(steps).rjust(3) + f'  Reward: {colour}' + f"{reward:+.1f}" + '\033[0m')


if __name__ == "__main__":
    env = gym.make('FrozenLake-v0', is_slippery=False)
    
    # Currently, memory growth needs to be the same across GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    dqn_solver = DQNSolver()

    print("\n\n" + '\033[92m' + "Begin training OpenAI Gym Frozen Lake" + '\033[0m' + "\n")

    for episode in range(EPISODES):
        state = env.reset()
        steps = 0
        done = False
        while not done:
            steps += 1
            action = dqn_solver.act(state)
            new_state, reward, done, _ = env.step(action)
            reward = -1.0 if done and reward < 1 else reward  
            dqn_solver.remember(state, action, reward, new_state, done) 
            dqn_solver.q_update(state, action, reward, new_state, done) 
            state = new_state
            if done:
                dqn_solver.report(episode, steps, reward)
                dqn_solver.epsilon_decay()
                dqn_solver.experience_replay()