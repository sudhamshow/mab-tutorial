import streamlit as st
from bandit import bernoulli_thompson_sampling

def main():
    st.title("Bernoulli Thompson Sampling")

    # Sidebar inputs
    num_arms = st.sidebar.slider("Number of Arms", min_value=2, max_value=10, value=3)
    num_iterations = st.sidebar.slider("Number of Iterations", min_value=100, max_value=10000, step=100, value=1000)

    # Generate arm probabilities
    probabilities = []
    st.sidebar.subheader("Arm Probabilities")
    for i in range(num_arms):
        probability = st.sidebar.slider(f"Arm {i+1} Probability", min_value=0.1, max_value=0.9, value=0.5)
        probabilities.append(probability)

    # Run Bernoulli Thompson Sampling
    selected_arms, cumulative_rewards = bernoulli_thompson_sampling(probabilities, num_iterations)

    # Display results
    st.header("Results")

    # Show the selected arms in a plot
    st.subheader("Selected Arms")
    st.line_chart(selected_arms)

    # Show the cumulative rewards in a plot
    st.subheader("Cumulative Rewards")
    st.line_chart(cumulative_rewards)

if __name__ == "__main__":
    main()
