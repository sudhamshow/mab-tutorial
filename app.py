import streamlit as st
import numpy as np
from bandit import EpsilonGreedy
from bandit import UCB

def run_simulation(bandit, true_rewards, num_iterations, arm_chart, prob_chart, selection_prob_chart, regret_chart):
    rewards = np.zeros(num_iterations)
    cumulative_regrets = np.zeros(num_iterations)
    average_regrets = np.zeros(num_iterations)
    selection_probabilities = np.zeros((num_iterations, bandit.num_arms))
    optimal_arm = np.argmax(true_rewards)
    for i in range(num_iterations):
        arm = bandit.select_arm()
        selection_probabilities[i, :] = bandit.arm_counts / (np.sum(bandit.arm_counts) + 1e-10)
        reward = np.random.binomial(1, true_rewards[arm])  # binary reward
        bandit.update(arm, reward)
        rewards[i] = reward
        cumulative_regrets[i] = (i+1) * true_rewards[optimal_arm] - np.sum(rewards)
        average_regrets[i] = cumulative_regrets[i] / (i+1)
        if (i+1) % 100 == 0:  # update every 100 iterations
            arm_chart.bar_chart(bandit.arm_counts / (i + 1))
            regret_chart.line_chart({'Cumulative Regret': cumulative_regrets[:i + 1],
                                     'Average Regret': average_regrets[:i + 1]})
            prob_chart.bar_chart(bandit.arm_rewards)
            selection_prob_chart.line_chart(selection_probabilities[:i + 1])
    return rewards, cumulative_regrets, average_regrets


# def plot_regret(cumulative_regrets, num_iterations):
#     fig, ax = plt.subplots()
#     ax.plot(cumulative_regrets, label="Cumulative Regret")
#     ax.plot(cumulative_regrets / np.arange(1, num_iterations+1), label="Average Regret")
#     ax.legend()
#     ax.set_xlabel('Iteration')
#     ax.set_ylabel('Regret')
#     ax.set_title('Average and Cumulative Regret')
#     return fig

def main():
    st.title("Multi-Armed Bandits Tutorial")

    st.markdown("""Author: *Sudhamshu Hosamane*""")

    st.sidebar.title("Index")
    options = st.sidebar.radio("Navigate to", ['Introduction', 'Epsilon-Greedy Algorithm', 'UCB Algorithm'])
    #
    # # Introduction
    if options == 'Introduction':
        st.header("Introduction")

        st.write("Imagine you walk into a casino and see a row of slot machines, each with a different lever or button to pull. You want to make as much money as possible, but you don't know which machines are the most likely to pay out. More formally, suppose there's a true distribution of rewards for each of the slot machines (which you are unaware of), and you need to come up with a strategy to maximize your potential winnings. This is the essence of the multi-armed bandits problem.")

        st.write("In real-world scenarios, multi-armed bandits have numerous applications where the problem is set up as previously defined. For example, in online advertising, you want to determine which ad to display to users to maximize click-through rates. Each ad can be considered an 'arm', and the goal is to learn which ad has the highest click-through rate by exploring and exploiting different options.")

        st.write("Another scenario is in clinical trials, where researchers need to identify the most effective treatment from a set of options. By applying multi-armed bandits, they can allocate patients to different treatments and adapt their strategy based on the observed outcomes. This adaptive approach allows for efficient exploration of different treatments and maximizes the chances of identifying the best one.")

        st.subheader("Exploration vs. Exploitation")

        st.markdown("""
        The multi-armed bandit problem is a classic dilemma in decision-making under uncertainty. Recall that your goal was to maximize the cumulative reward over a series of sequential actions (draws from the slot machine). You can achieve this while simultaneously balancing the exploration of potentially better options and the exploitation of known good options.
    
        **Exploration** involves trying out different arms to gather information about their reward probabilities. It allows the agent to discover potentially better arms.
    
        **Exploitation** involves choosing the arm that is estimated to have the highest reward probability based on the information collected so far. It maximizes the immediate reward.
    
        The multi-armed bandit problem aims to find the optimal tradeoff between exploration and exploitation to maximize the cumulative reward over time.
    
        There are two main categories of algorithms used to solve the multi-armed bandit problem:
    
        - **Frequentist Approaches**: These algorithms use Frequentist statistics to estimate the arm probabilities and make decisions based on past observations. Two popular frequentist algorithms are the *epsilon-greedy* and *Upper Confidence Bound (UCB)* methods.
    
        - **Bayesian Approaches**: These algorithms use Bayesian statistics (updating prior beliefs about the distributions) to model the uncertainty of arm probabilities and update their beliefs based on new observations. One commonly used Bayesian algorithm is *Bernoulli Thompson Sampling*.
    
        In this tutorial, we will delve into these algorithms and explore how they tackle the exploration-exploitation tradeoff to solve the multi-armed bandit problem.
        """)

        st.subheader("Classification of Multi-Armed Bandit Problems", anchor=True)

        st.write(
            "The Multi-Armed Bandit problem can be classified into two major types: Context-Free and Contextual. The main difference between these two types is the presence or absence of additional information (context) that can assist in making decisions.")

        st.write(
            "**Context-Free Bandits**: In this type of problem, each action (arm) has a fixed but unknown reward distribution. The only information available is the history of actions and rewards. The goal is to identify and exploit the arm with the highest expected reward. This is the classic formulation of the Multi-Armed Bandit problem. Previously mentioned algorithms (Epsilon-Greedy, UCB, Bernoulli Thompson Sampling) fall under this category.")

        st.write(
            "**Contextual Bandits**: In Contextual Bandits, also known as Associative Bandits, additional context is available to aid in decision making. Each action is accompanied by a d-dimensional context vector that provides more information about the action or the environment. The reward distributions are therefore not only dependent on the actions but also on the context. The goal in this case is to find a policy that uses the context to decide which arm to pull in order to maximize rewards.")

        st.write(
            "The Contextual Bandit problem is more complex but also more powerful as it allows us to make more informed decisions by incorporating additional information. This type of problem is more relevant in real-world applications like personalized recommendations, where the context could be user information or other environmental factors.")

        st.write("Before we proceed to study any specific algorithm, let us study a general framework to evaluate these multi-armed bandit algorithms")
        # Evaluation Metrics: Average Cumulative Regret and Simple Regret
        st.subheader(
            "Evaluation Metrics for Multi-Armed Bandit algorithms: Average Cumulative Regret and Simple Regret")

        st.write(
            "When evaluating multi-armed bandit algorithms, we need metrics to measure their performance. Two commonly used metrics are Average Cumulative Regret and Simple Regret.")

        st.markdown("""#### Simple Regret""")

        st.write(
            "Simple Regret, also known as Instantaneous Regret, measures the difference between the expected reward of the optimal arm and the reward obtained from the chosen arm at each time step t.")

        st.write("Mathematically, Simple Regret at time t can be defined as:")
        st.latex(r"S(t) = \max_{a^\ast \in \mathcal{A}} r^\ast - R(A)_{(t)}")

        st.write("where:")
        st.latex(r"S(t) = \text{Simple Regret at time t}")
        st.latex(
            r"a^\ast = \text{Optimal arm with the highest reward probability}")
        st.latex(r"r^\ast = \text{Expected reward of the optimal arm}")
        st.latex(
            r"R(A)_{(t)} = \text{Reward from the chosen arm (A) at time t}")

        st.write(
            "Simple Regret provides a measure of how much reward is lost at each time step by not selecting the optimal arm.")

        st.markdown("""#### Average Cumulative Regret""")

        st.write(
            "While Simple Regret captures the regret at each time step, Average Cumulative Regret sums up the regrets over a sequence of trials and provides a more comprehensive measure of the algorithm's performance.")

        st.write(
            "Mathematically, Average Cumulative Regret after T trials can be defined as:")
        st.latex(r"\mathcal{L}(T) = T r^\ast - \sum_{t=1}^{T} R(A)_{(t)}")

        st.write("where:")
        st.latex(r"\mathcal{L}(T) = \text{Average Cumulative Regret after T trials}")
        st.latex(r"T = \text{Number of trials}")
        st.latex(r"r^\ast = \text{Expected reward of the optimal arm}")
        st.latex(
            r"R(A)_{(t)} = \text{Reward from the chosen arm at time t}")

        st.write(
            "Average Cumulative Regret measures the total regret accumulated over T trials compared to always selecting the optimal arm. A lower Average Cumulative Regret indicates better performance.")

        st.write(
            "By evaluating multi-armed bandit algorithms using these metrics, we can compare their effectiveness in maximizing rewards and minimizing regret.")

    # Epsilon-Greedy Algorithm
    elif options == 'Epsilon-Greedy Algorithm':
        st.header("Epsilon-Greedy Algorithm")

        st.write("The Epsilon-Greedy algorithm is one of the most straightforward approaches to balancing the exploration-exploitation tradeoff. The algorithm maintains estimates of the expected rewards for each arm, primarily exploits the arm with the highest estimated reward (greedy action), but also explores other arms with a small probability epsilon.")

        st.write("To understand the working of the Epsilon-Greedy algorithm, consider the following step-by-step explanation:")

        st.write("1. Initialization: Initialize the estimated reward `R(a)` and the number of times each arm `a` has been selected `N(a)` for all arms.")

        st.write("2. Exploration: With a small probability `epsilon`, select a random arm. This action allows the algorithm to explore all arms and gather information about their actual reward probabilities.")

        st.write("3. Exploitation: If not exploring, the algorithm selects the arm with the highest estimated reward. This action is known as exploitation, where the algorithm makes the best decision based on the current knowledge.")

        st.write("4. Update: After receiving the reward from the chosen arm, the algorithm updates the estimated reward and selection count for that arm.")


        st.write("The Epsilon-Greedy algorithm is simple to implement and effective in scenarios with relatively stable reward distributions. Here's a basic implementation:")

        st.code("""
            class EpsilonGreedy:
    def __init__(self, num_arms, epsilon, adaptive=False):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.adaptive = adaptive
        self.arm_rewards = np.zeros(num_arms)
        self.arm_counts = np.zeros(num_arms)
    
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
            """, language='python')

        st.write("This code defines a class `EpsilonGreedy` that maintains the count and total reward for each arm. The `select_arm` method chooses an arm to play: with probability `epsilon`, it selects a random arm; otherwise, it selects the arm with the highest average reward. The `update` method updates the count and total reward for the chosen arm based on the received reward.")
        st.markdown("##### Vary the below parameters and observe they affect the probability of the selection of each arm and the Cumulative Regret. The rewards follow a Bernoulli distribution (binary reward) with true parameters adjustable below")

        st.markdown(r"""
                ##### Some key principles that are verifiable -
                * After a large number of iterations, the Epsilon-Greedy algorithm always converges to the most optimum arm, no matter how close the true means of the different arms are
                * When the true means for different arms are close to each other, the cumulative regret might see some negative values. But eventually after a large number of iterations, the cumulative regret always stays positive
                * There is a stark contrast in the growth rate of cumulative regret between *fixed* epsilon and *adaptive* epsilon. This is because the theoretical bound for the former is O(n) while the theoretical bound for the growth of the latter is O{sqrt(n)log(n)). The latter is a much slower growth than the former
                * The estimated rewards for each arm reach very close to the true means of the reward distributions
            """)
        num_arms = st.selectbox("Choose the number of arms", list(range(2, 6)))
        true_rewards = [st.slider(f"Set the true reward for arm {i}", 0.0, 1.0) for i in range(num_arms)]
        num_iterations = st.slider("Choose the number of iterations", 100, 50000)
        epsilon = st.slider("Choose the value of epsilon", 0.0, 1.0)
        epsilon_choice = st.selectbox("Choose the type of epsilon", ['Fixed', 'Adaptive'])

        st.subheader('Proportion of Observations Assigned to Each Arm')
        arm_chart = st.empty()
        # st.subheader('Selection Probabilities Over Time')
        st.subheader('Cumulative and Average Regret Over Time')
        selection_prob_chart = st.empty()
        st.subheader('Estimated Reward for each arm')
        regret_chart = st.empty()
        st.subheader('Selection Probabilities Over Time')
        prob_chart = st.empty()


        adaptive = (epsilon_choice == 'Adaptive')
        bandit = EpsilonGreedy(num_arms, epsilon, adaptive)
        rewards, cumulative_regrets, average_regrets = run_simulation(bandit, true_rewards, num_iterations, arm_chart,
                                                                      regret_chart, prob_chart, selection_prob_chart)

    elif options == 'UCB Algorithm':
        # Upper Confidence Bound (UCB) Algorithm
        st.header("Upper Confidence Bound (UCB) Algorithm")
        st.write(
            "The Upper Confidence Bound (UCB) algorithm uses the principle of optimism in the face of uncertainty to balance exploration and exploitation.")

        st.write(
            "The algorithm maintains upper confidence bounds for the expected rewards of each arm. It selects the arm with the highest upper confidence bound, which takes into account both the estimated reward and the uncertainty.")

        st.write("Mathematically, let's define:")
        st.latex(r"\overline{R}(a) = \frac{1}{N(a)} \sum_{t=1}^{N(a)} R_{t}(a)")
        st.latex(r"UCB(a) = \overline{R}(a) + \sqrt{\frac{2 \log(N)}{N(a)}}")

        st.write("The algorithm works as follows:")
        st.write("- Initialize N(a) and select each arm once")
        st.write("- For each subsequent selection:")
        st.write("    - Calculate the upper confidence bound (UCB) for each arm")
        st.write("    - Select the arm with the highest UCB (exploitation)")
        st.write("    - Update the selection count for the chosen arm")
        st.write(
            "    - Update the estimated reward for the chosen arm based on the observed reward")

        st.write(
            "The UCB algorithm adapts to the observed rewards and is particularly useful when dealing with uncertain or non-stationary reward distributions.")
        num_arms = st.selectbox("Choose the number of arms", list(range(2, 6)))
        true_rewards = [st.slider(f"Set the true reward for arm {i}", 0.0, 1.0) for i in range(num_arms)]
        num_iterations = st.slider("Choose the number of iterations", 100, 50000)

        st.subheader('Proportion of Observations Assigned to Each Arm')
        arm_chart = st.empty()
        st.subheader('Estimated Reward for each arm')
        regret_chart = st.empty()
        st.subheader('Selection Probabilities Over Time')
        prob_chart = st.empty()
        st.subheader('Cumulative and Average Regret Over Time')
        selection_prob_chart = st.empty()

        bandit = UCB(num_arms)
        rewards, cumulative_regrets, average_regrets = run_simulation(bandit, true_rewards, num_iterations, arm_chart,
                                                                      regret_chart, prob_chart, selection_prob_chart)


if __name__ == "__main__":
    main()
