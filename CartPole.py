# **
# * Author:    Chris Ryan
# * Created:   14.05.2021
# ** 
# A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The 
# pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends
# when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

# Hyperparameters (Bellman equation)
# α (alpha)   The learning rate (0.001). Dictates how much existing best known values are impacted by
# γ (gamma)   The reward discount factor. Normally between 0.90-0.99). Favours shorter-term rewards
# ε (epsilon) Governs the balance between exploratory (random) action and greedy (best known) action.
#             This needs to decrease over time to favour best known action.

import random, time
import gym
import numpy as np
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers

ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1_000_000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

# epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

class DQNSolver:

    def __init__(self, observation_space, action_space):
        self.epsilon = EXPLORATION_MAX

        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        self.model = keras.Sequential()
        self.model.add(layers.Dense(24, input_shape=(observation_space,), activation="relu"))
        self.model.add(layers.Dense(24, activation="relu"))
        self.model.add(layers.Dense(self.action_space, activation="linear"))
        self.model.compile(loss="mse", optimizer=optimizers.Adam(lr=LEARNING_RATE))
        self.model.summary()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in batch:
            q_update = reward
            if not done:
                q_update = (reward + GAMMA * np.amax(self.model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        if self.epsilon > EXPLORATION_MIN:
            self.epsilon *= EXPLORATION_DECAY

def cartpole():
    env = gym.make(ENV_NAME)
    observation_space = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNSolver(observation_space, action_space)
    run = 0
    while True:
        run += 1 
        state = env.reset()
        state = np.reshape(state, [1, observation_space])
        print(state)
        step = 0
        while True:
            step += 1
            action = dqn_solver.act(state)
            state_next, reward, done, info = env.step(action)
            x, x_dot, theta, theta_dot = state_next
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            newreward = r1 + r2
            reward = newreward if not done else -1.0
            state_next = np.reshape(state_next, [1, observation_space])
            dqn_solver.remember(state, action, reward, state_next, done)
            state = state_next
            if done:
                print('Run: ' + str(run) + ' Epsilon: {:.2f}'.format(dqn_solver.epsilon) + ' Score: ' + str(step))
                break
            dqn_solver.experience_replay()

if __name__ == "__main__":
    cartpole()
