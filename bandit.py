import numpy as np
import math
from scipy.stats import beta

class BernoulliThompson:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms) # choosing a flat prior
        self.beta = np.ones(num_arms)   # akin to sampling uniformly randomly
        self.arm_counts = np.zeros(num_arms)  # Number of times each arm is pulled
        self.total_count = 0  # Total number of pulls

    def select_arm(self):
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.num_arms)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        self.arm_counts[chosen_arm] += 1
        self.total_count += 1

class EpsilonGreedy:
    def __init__(self, num_arms, epsilon, adaptive=False):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.adaptive = adaptive
        self.arm_rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
        self.total_count = 0

    def select_arm(self):
        epsilon = self.epsilon
        if self.adaptive:
            if self.total_count > 0:
                epsilon = 1.0 / (self.total_count ** (1/2))

        if np.random.rand() < epsilon:
            return np.random.choice(self.num_arms) # Exploration
        else:
            return np.argmax(self.arm_rewards) # Exploitation

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.arm_counts[chosen_arm] += 1
        count = self.arm_counts[chosen_arm]
        value = self.arm_rewards[chosen_arm]
        new_value = ((count - 1) / float(count)) * value + (1 / float(count)) * reward
        self.arm_rewards[chosen_arm] = new_value

    def reset(self):
        self.total_count = 0
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms)


class UCB:
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.arm_rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
        self.ucbs = np.zeros(num_arms)  # UCB values for each arm
        self.total_count = 0

    def select_arm(self):
        if self.total_count < self.num_arms:
            return self.total_count
        else:
            ucb_values = self.arm_rewards + np.sqrt((2 * np.log(self.total_count)) / (self.arm_counts + 1e-10))
            self.ucbs = ucb_values
            return np.argmax(ucb_values)

    def update(self, chosen_arm, reward):
        self.total_count += 1
        self.arm_counts[chosen_arm] += 1
        count = self.arm_counts[chosen_arm]
        value = self.arm_rewards[chosen_arm]
        new_value = ((count - 1) / float(count)) * value + (1 / float(count)) * reward
        self.arm_rewards[chosen_arm] = new_value

    def reset(self):
        self.total_count = 0
        self.arm_rewards = np.zeros(self.num_arms)
        self.arm_counts = np.zeros(self.num_arms)