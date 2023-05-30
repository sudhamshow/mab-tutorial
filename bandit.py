import numpy as np
import math
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
        self.total_count = 0

    def select_arm(self):
        if self.total_count < self.num_arms:
            return self.total_count
        else:
            return np.argmax(self.arm_rewards + np.sqrt((2 * np.log(self.total_count)) / (self.arm_counts)))

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