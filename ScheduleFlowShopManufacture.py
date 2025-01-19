import streamlit as st
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

# Load dataset function
def load_data(file):
    data = flowshop_scheduling_dataset.csv(file)
    job_dict = data.set_index('Job_ID').to_dict('index')
    return data, job_dict

# Fitness function
def calculate_fitnessMO(route, job_dict):
    num_jobs = len(route)
    num_machines = 2
    completion_times = np.zeros((num_jobs, num_machines))
    total_penalty = 0

    for i, job_id in enumerate(route):
        proc_time_1 = job_dict[job_id]['Processing_Time_Machine_1']
        proc_time_2 = job_dict[job_id]['Processing_Time_Machine_2']
        setup_time_1 = job_dict[job_id]['Setup_Time_Machine_1']
        setup_time_2 = job_dict[job_id]['Setup_Time_Machine_2']
        due_date = job_dict[job_id]['Due_Date']
        weight = job_dict[job_id]['Weight']

        if i == 0:
            completion_times[i, 0] = proc_time_1 + setup_time_1
            completion_times[i, 1] = completion_times[i, 0] + proc_time_2 + setup_time_2
        else:
            completion_times[i, 0] = completion_times[i - 1, 0] + proc_time_1 + setup_time_1
            completion_times[i, 1] = max(completion_times[i, 0], completion_times[i - 1, 1]) + proc_time_2 + setup_time_2

        lateness = max(0, completion_times[i, 1] - due_date)
        total_penalty += weight * lateness

    makespan = completion_times[-1, -1]
    return makespan + total_penalty

# Evolutionary Strategies function
def evolutionary_strategies(data, job_dict, lambda_offspring, pop_size=50, mutation_rate=0.2, generations=100):
    jobs = list(data["Job_ID"])
    population = [random.sample(jobs, len(jobs)) for _ in range(pop_size)]
    fitness_trends = []

    best_fitness = float("inf")
    best_schedule = None

    for generation in range(generations):
        fitness = [calculate_fitnessMO(schedule, job_dict) for schedule in population]
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best_fitness = fitness[best_index]
            best_schedule = population[best_index]

        fitness_trends.append(best_fitness)

        offspring = []
        for _ in range(lambda_offspring):
            parent = random.choice(population)
            child = parent[:]
            idx1, idx2 = random.sample(range(len(child)), 2)
            child[idx1], child[idx2] = child[idx2], child[idx1]
            offspring.append(child)

        combined_population = population + offspring
        combined_fitness = fitness + [calculate_fitnessMO(child, job_dict) for child in offspring]
        selected_indices = np.argsort(combined_fitness)[:pop_size]
        population = [combined_population[i] for i in selected_indices]

    return best_schedule, fitness_trends

# Streamlit app
st.header("Evolutionary Strategies for Job Scheduling a Flow Shop Manufacturing", divider="gray")

# File uploader
data_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
if data_file is not None:
    data, job_dict = load_data(data_file)
    st.write("Dataset Preview:")
    st.dataframe(data.head())

    # Parameters
    lambda_offspring = st.slider("Lambda Offspring:", 10, 100, 35)
    pop_size = st.slider("Population Size:", 10, 100, 50)
    mutation_rate = st.slider("Mutation Rate:", 0.0, 1.0, 0.2, step=0.05)
    generations = st.slider("Generations:", 10, 200, 100)

    # Run algorithm
    if st.button("Run Evolutionary Strategies"):
        with st.spinner("Running Evolutionary Strategies..."):
            best_schedule, fitness_trends = evolutionary_strategies(
                data, job_dict, lambda_offspring, pop_size, mutation_rate, generations
            )

        st.success("Algorithm completed!")
        st.write("Best Job Schedule:")
        st.write(best_schedule)

        st.write("Best Fitness Value:", fitness_trends[-1])

        # Plot fitness trends
        st.write("Fitness Trends:")
        fig, ax = plt.subplots()
        ax.plot(range(1, len(fitness_trends) + 1), fitness_trends, marker='o', linestyle='-', color='b')
        ax.set_title("Fitness Trends Over Generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Best Fitness")
        ax.grid(alpha=0.4)
        st.pyplot(fig)

st.markdown("---")
st.markdown("Developed with Streamlit and Python")
