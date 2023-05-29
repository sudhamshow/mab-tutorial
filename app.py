import streamlit as st

def main():
    st.title("Multi-Armed Bandits Tutorial")

    # Introduction to Multi-Armed Bandits
    st.header("Introduction")

    st.write("Imagine you walk into a casino and see a row of slot machines, each with a different lever or button to pull. You want to make as much money as possible, but you don't know which machines are the most likely to pay out. More formally, suppose there's a true distribution of rewards for each of the slot machines (which you are unaware of), and you need to come up with a strategy to maximize your potential winnings. This is the essence of the multi-armed bandits problem.", fontsize=40)

    st.write("In real-world scenarios, multi-armed bandits have numerous applications where the problem is set up as previously defined. For example, in online advertising, you want to determine which ad to display to users to maximize click-through rates. Each ad can be considered an 'arm', and the goal is to learn which ad has the highest click-through rate by exploring and exploiting different options.")

    st.write("Another scenario is in clinical trials, where researchers need to identify the most effective treatment from a set of options. By applying multi-armed bandits, they can allocate patients to different treatments and adapt their strategy based on the observed outcomes. This adaptive approach allows for efficient exploration of different treatments and maximizes the chances of identifying the best one.")

    st.subheader("Exploration vs. Exploitation")

    st.markdown("""
    The multi-armed bandit problem is a classic dilemma in decision-making under uncertainty. It refers to a scenario where an agent (or gambler) is faced with a set of slot machines (or bandits), each with a different unknown probability distribution of rewards. The agent's goal is to maximize its cumulative reward over a series of sequential actions, while simultaneously balancing the exploration of potentially better options and the exploitation of known good options.

    **Exploration** involves trying out different arms to gather information about their reward probabilities. It allows the agent to discover potentially better arms.

    **Exploitation** involves choosing the arm that is estimated to have the highest reward probability based on the information collected so far. It maximizes the immediate reward.

    The multi-armed bandit problem aims to find the optimal tradeoff between exploration and exploitation to maximize the cumulative reward over time.

    There are two main categories of algorithms used to solve the multi-armed bandit problem:

    - **Frequentist Approaches**: These algorithms use frequentist statistics to estimate the arm probabilities and make decisions based on past observations. Two popular frequentist algorithms are the *epsilon-greedy* and *Upper Confidence Bound (UCB)* methods.

    - **Bayesian Approaches**: These algorithms use Bayesian statistics to model the uncertainty of arm probabilities and update their beliefs based on new observations. One commonly used Bayesian algorithm is *Bernoulli Thompson Sampling*.

    In this tutorial, we will delve into these algorithms and explore how they tackle the exploration-exploitation tradeoff to solve the multi-armed bandit problem.
    """)

    # Epsilon-Greedy Algorithm
    st.header("Epsilon-Greedy Algorithm")
    st.write("The Epsilon-Greedy algorithm is a classic and straightforward approach to balancing exploration and exploitation.")

    st.write("The algorithm maintains estimates of the expected rewards for each arm. It primarily exploits the arm with the highest estimated reward (greedy action) but also explores other arms with a small probability epsilon.")

    st.write("Mathematically, let's define:")
    st.latex(r"Q(a) = \text{{Estimated reward of arm }} a")
    st.latex(r"N(a) = \text{{Number of times arm }} a \text{{ has been selected}}")

    st.write("The algorithm works as follows:")
    st.write("- Initialize Q(a) and N(a) for each arm a")
    st.write("- With probability epsilon, select a random arm (exploration)")
    st.write("- Otherwise, select the arm with the highest estimated reward (exploitation)")
    st.write("- Update the estimated reward and selection count for the chosen arm based on the observed reward")

    st.write("The Epsilon-Greedy algorithm is simple to implement and effective in scenarios with relatively stable reward distributions.")

    # Upper Confidence Bound (UCB) Algorithm
    st.header("Upper Confidence Bound (UCB) Algorithm")
    st.write("The Upper Confidence Bound (UCB) algorithm uses the principle of optimism in the face of uncertainty to balance exploration and exploitation.")

    st.write("The algorithm maintains upper confidence bounds for the expected rewards of each arm. It selects the arm with the highest upper confidence bound, which takes into account both the estimated reward and the uncertainty.")

    st.write("Mathematically, let's define:")
    st.latex(r"\overline{X}(a) = \frac{1}{N(a)} \sum_{t=1}^{N(a)} X_{t}(a)")
    st.latex(r"UCB(a) = \overline{X}(a) + \sqrt{\frac{2 \log(N)}{N(a)}}")

    st.write("The algorithm works as follows:")
    st.write("- Initialize N(a) and select each arm once")
    st.write("- For each subsequent selection:")
    st.write("    - Calculate the upper confidence bound (UCB) for each arm")
    st.write("    - Select the arm with the highest UCB (exploitation)")
    st.write("    - Update the selection count for the chosen arm")
    st.write("    - Update the estimated reward for the chosen arm based on the observed reward")

    st.write("The UCB algorithm adapts to the observed rewards and is particularly useful when dealing with uncertain or non-stationary reward distributions.")

    # Bernoulli Thompson Sampling Algorithm
    st.header("Bernoulli Thompson Sampling Algorithm")
    st.write("The Bernoulli Thompson Sampling algorithm, also known as Bayesian Bandit, leverages Bayesian inference to estimate arm probabilities.")

    st.write("The algorithm starts with prior distributions over the arm probabilities and updates them based on the observed rewards. In each iteration, it samples from the posterior distributions and selects the arm with the highest sample.")

    st.write("Mathematically, let's define:")
    st.latex(r"\text{{Beta}}(\alpha, \beta) = \text{{Beta distribution with parameters }} \alpha \text{{ and }} \beta")
    st.latex(r"\text{{Beta}}(\alpha + R(a), \beta + N(a) - R(a)) = \text{{Posterior distribution of arm }} a")

    st.write("The algorithm works as follows:")
    st.write("- Initialize the prior distributions for each arm")
    st.write("- For each selection:")
    st.write("    - Sample from the posterior distribution of each arm")
    st.write("    - Select the arm with the highest sample (exploitation)")
    st.write("    - Update the observed reward and selection count for the chosen arm")

    st.write("Bernoulli Thompson Sampling adapts to sparse and non-stationary reward distributions, making informed decisions based on posterior beliefs.")

    # Evaluation Metrics: Average Cumulative Regret and Simple Regret
    st.header("Evaluation Metrics for Multi-Armed Bandit algorithms: Average Cumulative Regret and Simple Regret")

    st.write("When evaluating multi-armed bandit algorithms, we need metrics to measure their performance. Two commonly used metrics are Average Cumulative Regret and Simple Regret.")

    st.subheader("Simple Regret")

    st.write("Simple Regret, also known as Instantaneous Regret, measures the difference between the expected reward of the optimal arm and the reward obtained from the chosen arm at each time step t.")

    st.write("Mathematically, Simple Regret at time t can be defined as:")
    st.latex(r"S(t) = \max_{a^\ast \in \mathcal{A}} \mu^\ast - \mu_{A(t)}")

    st.write("where:")
    st.latex(r"S(t) = \text{Simple Regret at time t}")
    st.latex(r"a^\ast = \text{Optimal arm with the highest reward probability}")
    st.latex(r"\mu^\ast = \text{Expected reward of the optimal arm}")
    st.latex(r"\mu_{A(t)} = \text{Expected reward of the chosen arm at time t}")

    st.write("Simple Regret provides a measure of how much reward is lost at each time step by not selecting the optimal arm.")

    st.subheader("Average Cumulative Regret")

    st.write("While Simple Regret captures the regret at each time step, Average Cumulative Regret sums up the regrets over a sequence of trials and provides a more comprehensive measure of the algorithm's performance.")

    st.write("Mathematically, Average Cumulative Regret after T trials can be defined as:")
    st.latex(r"R(T) = T \mu^\ast - \sum_{t=1}^{T} \mu_{A(t)}")

    st.write("where:")
    st.latex(r"R(T) = \text{Average Cumulative Regret after T trials}")
    st.latex(r"T = \text{Number of trials}")
    st.latex(r"\mu^\ast = \text{Expected reward of the optimal arm}")
    st.latex(r"\mu_{A(t)} = \text{Expected reward of the chosen arm at time t}")

    st.write("Average Cumulative Regret measures the total regret accumulated over T trials compared to always selecting the optimal arm. A lower Average Cumulative Regret indicates better performance.")

    st.write("By evaluating multi-armed bandit algorithms using these metrics, we can compare their effectiveness in maximizing rewards and minimizing regret.")

    # Adaptive Experiments and Policy Interventions
    st.header("Adaptive Experiments and Policy Interventions")
    st.write("Adaptive experiments refer to experiments where the allocation of treatments or interventions adapts based on the observed outcomes. The goal is to identify the best policy or treatment from a set of options.")

    st.write("In adaptive experiments, policies are often represented as probability distributions over the arms. Each policy specifies the probability of selecting each arm at any given time.")

    st.write("Mathematically, let's define:")
    st.latex(r"\pi = (\pi_1, \pi_2, ..., \pi_k) = \text{{Probability distribution over the arms}}")
    st.latex(r"T(a) = \text{{Number of times arm }} a \text{{ has been selected}}")
    st.latex(r"R(a) = \text{{Sum of rewards obtained from arm }} a")
    st.latex(r"\hat{r}(a) = \frac{R(a)}{T(a)} = \text{{Estimated reward of arm }} a")

    st.write("The objective is to maximize the cumulative reward or minimize the cumulative regret over a series of trials.")

    st.write("One commonly used approach is to select the policy with the highest upper confidence bound (UCB) estimate. The UCB estimate for each policy is calculated as:")
    st.latex(r"UCB(a) = \hat{r}(a) + \sqrt{\frac{2 \log(n)}{T(a)}}")

    st.write("By choosing the policy with the highest UCB, we balance exploration (choosing arms with high uncertainty) and exploitation (choosing arms with high estimated rewards). This adaptive approach allows us to learn and adapt our policy based on the observed outcomes.")

    st.write("Adaptive experiments have been successfully applied in various policy interventions. For example, in a public health campaign, policymakers can allocate resources to different interventions and adapt their strategy based on the observed impact on desired outcomes. By dynamically allocating resources based on observed data, policymakers can make informed decisions and optimize the effectiveness of their interventions.")

if __name__ == "__main__":
	main()
