import streamlit as st
from bandit import bernoulli_thompson_sampling

def main():
import streamlit as st

def main():
    st.title("Multi-Armed Bandits Tutorial")

    # Introduction to Multi-Armed Bandits
    st.header("Introduction")
    st.write("Welcome to the Multi-Armed Bandits Tutorial! Get ready to explore the fascinating world of decision-making under uncertainty.")

    st.write("Imagine you walk into a casino and see a row of slot machines, each with a different lever or button to pull. Your goal is to maximize your winnings over a series of trials. However, you don't know the exact reward probabilities of each machine. This is the essence of the multi-armed bandits problem.")

    st.write("In real-world scenarios, multi-armed bandits have numerous applications. For example, in online advertising, you want to determine which ad to display to users to maximize click-through rates. Each ad can be considered an 'arm', and the goal is to learn which ad has the highest click-through rate by exploring and exploiting different options.")

    st.write("Another scenario is in clinical trials, where researchers need to identify the most effective treatment from a set of options. By applying multi-armed bandits, they can allocate patients to different treatments and adapt their strategy based on the observed outcomes. This adaptive approach allows for efficient exploration of different treatments and maximizes the chances of identifying the best one.")

    st.write("Now, let's delve into the different algorithms used in multi-armed bandits.")

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

    # Adaptive Experiments and Policy Interventions
    st.header("Adaptive Experiments and Policy Interventions")
    st.write("Multi-armed bandits are not limited to casino games or online advertising. They have become increasingly popular in adaptive experiments and policy interventions.")

    st.write("Adaptive experiments refer to experiments where the allocation of treatments or interventions adapts based on the observed outcomes. By applying multi-armed bandit algorithms, researchers can efficiently explore different treatments and identify the most effective one.")

    st.write("In policy interventions, multi-armed bandits can help optimize resource allocation. For example, in a public health campaign, policymakers can allocate resources to different interventions and adapt their strategy based on the observed impact on desired outcomes.")

    st.write("These applications involve dynamically allocating resources based on observed data, leading to better decision-making and more efficient outcomes.")

    st.write("Now that you have a deeper understanding of multi-armed bandits, their algorithms, and their applications, you can apply them to various scenarios and make optimal decisions under uncertainty.")


if __name__ == "__main__":
    main()
