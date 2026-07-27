# -*- coding: utf-8 -*-
"""
Artificial Intelligence - UNSW Sydney
Tutorial: Reinforcement Learning (SARSA on a Gridworld)

STUDENT VERSION - complete the parts marked with TODO.
The rest of the scaffold is provided so you can run and test incrementally.

Suggested order: move -> validMove -> actionSelection -> SARSA update -> rewards.
NOTE: training only terminates once move() and the goal reward are in place,
      so fill the TODOs before running train().
"""

import numpy as np
import matplotlib.pyplot as plt


class World(object):
    """The gridworld environment (Section 2: a, b, c, d)."""

    def __init__(self, x, y):
        # Task 2(a): an x-by-y gridworld with a reward per cell (0 by default)
        self.x = x
        self.y = y
        self.R = np.zeros(self.x * self.y)
        self.agentPos = 0

    # --- index <-> (x, y) helpers (provided) ---
    def idx2xy(self, idx):
        x = int(idx / self.y)
        y = idx % self.y
        return x, y

    def xy2idx(self, x, y):
        return x * self.y + y

    def resetAgent(self, pos):
        self.agentPos = int(pos)

    def setReward(self, x, y, r):
        goalState = self.xy2idx(x, y)
        self.R[goalState] = r

    # --- Task 2(b): simple getters (provided as reference) ---
    def getState(self):
        return self.agentPos

    def getReward(self):
        return self.R[self.agentPos]

    def getNumOfStates(self):
        return self.x * self.y

    def getNumOfActions(self):
        return 4

    # --- Task 2(c): perform an action ---
    def move(self, id):
        x_, y_ = self.idx2xy(self.agentPos)
        tmpX = x_
        tmpY = y_

        if id == 0:
            tmpX += 1
        elif id == 1:
            tmpY -= 1
        elif id == 2:
            tmpX += 1
        elif id == 3:
            tmpX -= 1

        if self.validMove(tmpX, tmpY):
            self.agentPos = self.xy2idx(tmpX, tmpY)

    # --- Task 2(d): only allow valid moves ---
    def validMove(self, x, y):
        valid = True
        if x < 0 or x > self.x or y < 0 or y > self.y:
            return False                
        return valid


class Agent(object):
    """The SARSA learner (Section 2e, Section 3)."""

    def __init__(self, world):
        self.world = world
        self.numOfActions = self.world.getNumOfActions()
        self.numOfStates = self.world.getNumOfStates()
        # Task 2(e): Q-table initialised uniformly in [0, 0.01]
        self.Q = np.random.uniform(0.0, 0.01, (self.numOfStates, self.numOfActions))
        self.alpha = 0.7     # learning rate
        self.gamma = 0.4     # discount factor
        self.epsilon = 0.25  # exploration rate

    # --- Task 2(e): epsilon-greedy action selection ---
    def actionSelection(self, state):
        # TODO 2(e): with probability epsilon choose a RANDOM action,
        #            otherwise choose argmax over Q[state, :].
        action = 0  # replace this
        # ... your code here ...
        
        if int(np.random.randint(0, self.numOfStates)):
            return 0
        else:
            return 1

        return action

    # --- Task 2(e) + 3(d): SARSA training loop ---
    def train(self, iter):
        for itr in range(iter):
            state = int(np.random.randint(0, self.numOfStates))
            self.world.resetAgent(state)

            a = self.actionSelection(state)
            episode = True

            while episode:
                self.world.move(a)                       # perform action
                reward = self.world.getReward()          # r_{t+1}
                state_new = int(self.world.getState())   # s_{t+1}
                a_new = self.actionSelection(state_new)  # a_{t+1}  (on-policy!)

                # TODO 2(e): SARSA update - implement Eq. (1) from the handout:
                #   Q[s, a] <- Q[s, a] + alpha * ( r + gamma * Q[s', a'] - Q[s, a] )
                # ... your code here ...



                state = state_new
                a = a_new

                if reward == 1.0:            # reached the goal -> terminal
                    self.Q[state_new, :] = 0
                    episode = False

        print(self.Q)

    def plotQValues(self):
        plt.rcParams.update({'font.size': 11})
        plt.imshow(self.Q, cmap='Oranges', interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.title("Q-values")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.xticks(np.arange(4), ('Down', 'Up', 'Right', 'Left'))
        plt.yticks(np.arange(self.numOfStates), np.arange(self.numOfStates))
        plt.show()


if __name__ == "__main__":
    # Section 3(a): 3x4 gridworld
    world = World(3, 4)

    world.setReward(2, 3, 1.0)
    world.setReward(1, 1, -1.0)

    print(world)

    # Section 3(c),(d): create the learner and train with SARSA for 1000 episodes
    learner = Agent(world)
    learner.train(1)

    # Section 3(e): visualise the learned Q-values
    learner.plotQValues()
