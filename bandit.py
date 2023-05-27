import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def bernoulli_thompson_sampling(probabilities, num_iterations):
    num_arms = len(probabilities)
    successes = np.ones(num_arms)
    failures = np.ones(num_arms)
    selected_arms = []
    cumulative_rewards = []

    for _ in range(num_iterations):
        sampled_theta = np.random.beta(successes, failures)
        selected_arm = np.argmax(sampled_theta)
        reward = simulate_rewards(probabilities[selected_arm])
        successes[selected_arm] += reward
        failures[selected_arm] += 1 - reward

        selected_arms.append(selected_arm + 1)
        cumulative_rewards.append(sum(successes) / sum(successes + failures))

    return selected_arms, cumulative_rewards

def simulate_rewards(probability):
    return np.random.binomial(1, probability)

def main():
    # Initialize Streamlit application
    st.title("Bernoulli Thompson Sampling")

    # Sidebar inputs
    num_arms = st.sidebar.slider("Number of Arms", min_value=2, max_value=10, value=3)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=10000, step=100, value=1000)
    data_size = st.sidebar.slider("Data Size", min_value=10, max_value=1000, step=10, value=100)

    # Generate arm probabilities
    probabilities = np.random.uniform(low=0.1, high=0.9, size=num_arms)

    # Run Bernoulli Thompson Sampling
    selected_arms, cumulative_rewards = bernoulli_thompson_sampling(num_arms, num_iterations, probabilities)

    # Calculate regret
    optimal_arm = np.argmax(probabilities)
    regret = np.cumsum(probabilities[optimal_arm] - np.array([probabilities[selected_arm] for selected_arm in selected_arms]))

    # Display results
    st.header("Results")

    # Show the selected arms in a plot
    st.subheader("Selected Arms")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(selected_arms)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Selected Arm")
    st.pyplot(fig)

    # Show the cumulative rewards in a plot
    st.subheader("Cumulative Rewards")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cumulative_rewards)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Rewards")
    st.pyplot(fig)

    # Show the regret in a plot
    st.subheader("Regret")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(regret)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regret")
    st.pyplot(fig)

    # Data generation for large data size
    large_data_probabilities = np.random.uniform(low=0.1, high=0.9, size=num_arms)
    large_data_selected_arms, large_data_cumulative_rewards = bernoulli_thompson_sampling(num_arms, data_size, large_data_probabilities)

    # Show the selected arms for large data size in a plot
    st.subheader(f"Selected Arms for Data Size {data_size}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(large_data_selected_arms)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Selected Arm")
    st.pyplot(fig)

    # Show the cumulative rewards for large data size in a plot
    st.subheader(f"Cumulative Rewards for Data Size {data_size}")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(large_data_cumulative_rewards)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Rewards")
    st.pyplot(fig)

    # Theoretical limits and performance discussion
    st.header("Theoretical Limits and Performance")

    # Explain theoretical limits and practical performance considerations
    st.markdown("""
		The Bernoulli Thompson Sampling algorithm provides an effective approach to tackle the multi-armed bandit problem. However, it is important to understand its theoretical limits and practical considerations:

		- **Exploration vs. Exploitation:** Bernoulli Thompson Sampling balances exploration and exploitation by using Bayesian inference. It explores different arms initially but gradually focuses on the most rewarding arms based on observations.

		- **Optimal vs. Suboptimal Actions:** Bernoulli Thompson Sampling aims to identify the optimal action with the highest probability of success. However, it is possible to get stuck in suboptimal actions, especially if they were initially chosen due to chance.

		- **Data Size and Convergence:** The algorithm's performance improves with more iterations, allowing it to converge towards the optimal solution. However, the convergence speed depends on the data size and arm probabilities.

		- **Large Data Size:** Increasing the data size enhances the algorithm's performance and reduces the impact of initial randomness. In the plots above, you can compare the selected arms and cumulative rewards between the original data size and a larger data size.

		- **Inaccurate Prior Information:** Bernoulli Thompson Sampling assumes prior knowledge of arm probabilities. If the prior information is inaccurate, it may take longer to identify the optimal action.

		- **Contextual Bandits:** Bernoulli Thompson Sampling is a non-contextual bandit algorithm, which means it does not consider contextual information such as demographics. If your application involves contextual information, you may consider using contextual bandit algorithms instead.

		- **Real-World Applications:** Bernoulli Thompson Sampling has been successfully applied to various domains, including online advertising, recommendation systems, and clinical trials.

		By experimenting with different parameters and data sizes, you can gain insights into how Bernoulli Thompson Sampling performs under different scenarios and explore its theoretical limits.

		Feel free to adjust the sliders in the sidebar to explore different settings and observe the algorithm's behavior.
		""")


if __name__ == "__main__":
    main()
