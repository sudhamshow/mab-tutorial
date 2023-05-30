import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from bandit import EpsilonGreedy

def run_simulation(bandit, true_rewards, num_iterations, arm_chart, regret_chart, prob_chart):
	rewards = np.zeros(num_iterations)
	cumulative_regrets = np.zeros(num_iterations)
	optimal_arm = np.argmax(true_rewards)
	for i in range(num_iterations):
		arm = bandit.select_arm()
		reward = np.random.binomial(1, true_rewards[arm])  # binary reward
		bandit.update(arm, reward)
		rewards[i] = reward
		cumulative_regrets[i] = (i+1) * true_rewards[optimal_arm] - np.sum(rewards)
		if (i+1) % 100 == 0:  # update every 100 iterations
			arm_chart.bar_chart(bandit.arm_counts / (i+1))
			fig = plot_regret(cumulative_regrets[:i+1], i+1)
			regret_chart.pyplot(fig)
			prob_chart.bar_chart(bandit.arm_rewards)
			plt.close(fig)
	return rewards, cumulative_regrets


def plot_regret(cumulative_regrets, num_iterations):
	fig, ax = plt.subplots()
	ax.plot(cumulative_regrets, label="Cumulative Regret")
	ax.plot(cumulative_regrets / np.arange(1, num_iterations+1), label="Average Regret")
	ax.legend()
	ax.set_xlabel('Iteration')
	ax.set_ylabel('Regret')
	return fig

def main():
	st.title("Multi-Armed Bandits Tutorial")

	# Introduction to Multi-Armed Bandits
	st.header("Introduction")

	st.write("Imagine you walk into a casino and see a row of slot machines, each with a different lever or button to pull. You want to make as much money as possible, but you don't know which machines are the most likely to pay out. More formally, suppose there's a true distribution of rewards for each of the slot machines (which you are unaware of), and you need to come up with a strategy to maximize your potential winnings. This is the essence of the multi-armed bandits problem.")

	st.write("In real-world scenarios, multi-armed bandits have numerous applications where the problem is set up as previously defined. For example, in online advertising, you want to determine which ad to display to users to maximize click-through rates. Each ad can be considered an 'arm', and the goal is to learn which ad has the highest click-through rate by exploring and exploiting different options.")

	st.write("Another scenario is in clinical trials, where researchers need to identify the most effective treatment from a set of options. By applying multi-armed bandits, they can allocate patients to different treatments and adapt their strategy based on the observed outcomes. This adaptive approach allows for efficient exploration of different treatments and maximizes the chances of identifying the best one.")

	st.subheader("Exploration vs. Exploitation")

	st.markdown("""
	The multi-armed bandit problem is a classic dilemma in decision-making under uncertainty. Recall that your goal is to maximize the cumulative reward over a series of sequential actions (draws from the slot machine). You can achieve this while simultaneously balancing the exploration of potentially better options and the exploitation of known good options.

	**Exploration** involves trying out different arms to gather information about their reward probabilities. It allows the agent to discover potentially better arms.

	**Exploitation** involves choosing the arm that is estimated to have the highest reward probability based on the information collected so far. It maximizes the immediate reward.

	The multi-armed bandit problem aims to find the optimal tradeoff between exploration and exploitation to maximize the cumulative reward over time.

	There are two main categories of algorithms used to solve the multi-armed bandit problem:

	- **Frequentist Approaches**: These algorithms use frequentist statistics to estimate the arm probabilities and make decisions based on past observations. Two popular frequentist algorithms are the *epsilon-greedy* and *Upper Confidence Bound (UCB)* methods.

	- **Bayesian Approaches**: These algorithms use Bayesian statistics (updating prior beliefs about the distributions) to model the uncertainty of arm probabilities and update their beliefs based on new observations. One commonly used Bayesian algorithm is *Bernoulli Thompson Sampling*.

	In this tutorial, we will delve into these algorithms and explore how they tackle the exploration-exploitation tradeoff to solve the multi-armed bandit problem.
	""")

	# Epsilon-Greedy Algorithm
	st.header("Epsilon-Greedy Algorithm")
	st.write("The Epsilon-Greedy algorithm is a classic and straightforward approach to balancing exploration and exploitation.")

	st.write("The algorithm maintains estimates of the expected rewards for each arm. It primarily exploits the arm with the highest estimated reward (greedy action) but also explores other arms with a small probability epsilon.")

	st.write("Mathematically, let's define:")
	st.latex(r"Q(a) = \text{{Estimated reward of arm }} a")
	st.latex(
		r"N(a) = \text{{Number of times arm }} a \text{{ has been selected}}")

	st.write("The algorithm works as follows:")
	st.write("- Initialize Q(a) and N(a) for each arm a")
	st.write("- With probability epsilon, select a random arm (exploration)")
	st.write(
		"- Otherwise, select the arm with the highest estimated reward (exploitation)")
	st.write("- Update the estimated reward and selection count for the chosen arm based on the observed reward")

	st.write("The Epsilon-Greedy algorithm is simple to implement and effective in scenarios with relatively stable reward distributions.")

	num_arms = st.selectbox("Choose the number of arms", list(range(2, 6)))
	true_rewards = [st.slider(f"Set the true reward for arm {i+1}", 0.0, 1.0) for i in range(num_arms)]
	num_iterations = st.slider("Choose the number of iterations", 100, 10000)
	epsilon = st.slider("Choose the value of epsilon", 0.0, 1.0)
	epsilon_choice = st.selectbox("Choose the type of epsilon", ['Fixed', 'Adaptive'])

	arm_chart = st.empty()  # placeholder for arm chart
	regret_chart = st.empty()  # placeholder for regret chart
	prob_chart = st.empty()  # placeholder for probability chart

	adaptive = (epsilon_choice == 'Adaptive')
	bandit = EpsilonGreedy(num_arms, epsilon, adaptive)
	rewards, cumulative_regrets = run_simulation(bandit, true_rewards, num_iterations, arm_chart, regret_chart, prob_chart)


	st.write("Final rewards:")
	st.bar_chart(bandit.arm_rewards)

if __name__ == "__main__":
	main()
