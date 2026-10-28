import numpy as np
import random
from numpy import savetxt
import sys
import matplotlib.pyplot as plt


class QLearning:

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.rewards_per_episode = []
        self.actions_per_episode = []

    def select_action(self, state):
        rv = random.uniform(0, 1)
        if rv < self.epsilon:
            return self.env.action_space.sample() # Explore action space
        return np.argmax(self.q_table[state]) # Exploit learned values

    def train(self, filename, plotFile):
        self.rewards_per_episode = []
        self.actions_per_episode = []
        for i in range(1, self.episodes + 1):
            (state, _) = self.env.reset()
            rewards = 0
            done = False
            actions = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Atualiza a acao escolhida, mesmo quando o taxi escorrega.
                old_value = self.q_table[state, action]
                next_max = 0 if terminated else np.max(self.q_table[next_state])
                new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
                self.q_table[state, action] = new_value

                state = next_state
                actions = actions + 1
                rewards = rewards + reward
                done = terminated or truncated

            self.actions_per_episode.append(actions)
            self.rewards_per_episode.append(rewards)
            if i % 1000 == 0:
                sys.stdout.write("Episodes: " + str(i) + '\r')
                sys.stdout.flush()

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_dec)

        savetxt(filename, self.q_table, delimiter=',')
        if plotFile is not None:
            self.plotactions(plotFile, self.actions_per_episode)
        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.plot(actions_per_episode)
        plt.xlabel('Episodios')
        plt.ylabel('Quantidade de acoes')
        plt.title('Acoes por episodio - Q-Learning')
        plt.tight_layout()
        plt.savefig(str(plotFile) + '.jpg')
        plt.close()
