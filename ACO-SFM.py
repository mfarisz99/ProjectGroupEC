import streamlit as st
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# Load dataset function
def load_data():
    # Automatically load the dataset (provide the file path)
    file_path = "flowshop_scheduling_dataset.csv"  # Replace with your actual file path
    try:
        data = pd.read_csv(file_path)
        job_dict = data.set_index('Job_ID').to_dict('index')
        return data, job_dict
    except FileNotFoundError:
        st.error(f"Dataset not found at {file_path}. Please check the file path.")
        return None, None

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

# Ant Colony Optimization function
def ant_colony_optimization(data, job_dict, num_ants=10, num_iterations=100, alpha=1, beta=2, evaporation_rate=0.5):
    jobs = list(data["Job_ID"])
    num_jobs = len(jobs)

    # Initialize pheromone matrix
    pheromone = np.ones((num_jobs, num_jobs))
    best_route = None
    best_fitness = float("inf")

    for iteration in range(num_iterations):
        all_routes = []
        all_fitness = []

        # Simulate ants constructing solutions
        for ant in range(num_ants):
            route = []
            visited = set()
            current_job = random.choice(jobs)  # Start from a random job
            route.append(current_job)
            visited.add(current_job)

            while len(route) < num_jobs:
                probabilities = []
                for job in jobs:
                    if job not in visited:
                        # Calculate transition probability
                        pheromone_level = pheromone[jobs.index(current_job), jobs.index(job)]
                        heuristic = 1 / (job_dict[job]['Processing_Time_Machine_1'] + job_dict[job]['Processing_Time_Machine_2'])
                        probabilities.append((job, (pheromone_level ** alpha) * (heuristic ** beta)))
                # Normalize probabilities
                total = sum(p[1] for p in probabilities)
                probabilities = [(job, prob / total) for job, prob in probabilities]

                # Choose the next job based on probability
                next_job = random.choices(
                    population=[p[0] for p in probabilities],
                    weights=[p[1] for p in probabilities]
                )[0]
                route.append(next_job)
                visited.add(next_job)
                current_job = next_job

            # Calculate fitness of the route
            fitness = calculate_fitnessMO(route, job_dict)
            all_routes.append(route)
            all_fitness.append(fitness)

        # Update the best route and fitness
        min_fitness_index = np.argmin(all_fitness)
        if all_fitness[min_fitness_index] < best_fitness:
            best_fitness = all_fitness[min_fitness_index]
            best_route = all_routes[min_fitness_index]

        # Update pheromone levels
        for i in range(num_jobs):
            for j in range(num_jobs):
                pheromone[i, j] *= (1 - evaporation_rate)  # Evaporation
        for route, fitness in zip(all_routes, all_fitness):
            for i in range(len(route) - 1):
                start = jobs.index(route[i])
                end = jobs.index(route[i + 1])
                pheromone[start, end] += 1 / fitness  # Deposit pheromones based on fitness

        # Store the best fitness trend for plotting
        fitness_trends.append(best_fitness)

    return best_route, best_fitness

# Streamlit app
st.header("Ant Colony Optimization for Flow Shop Scheduling", divider="gray")

# Load dataset automatically
data, job_dict = load_data()
if data is not None:
    st.write("Dataset Preview:")
    st.dataframe(data)

     # Display fixed parameters
    st.write("**Fixed Parameters:**")
    st.write("Mutation Rate: 0.2")
    st.write("Population Size: 50")
    st.write("Generations: 100")

    # Run algorithm
    if st.button("Run Ant Colony Optimization"):
        with st.spinner("Running Ant Colony Optimization..."):
            best_schedule, best_fitness = ant_colony_optimization(
                data, job_dict, num_ants=10, num_iterations=100, alpha=1, beta=2, evaporation_rate=0.5
            )

        st.success("Algorithm completed!")
        st.write("**Best Job Schedule:**")
        st.write(best_schedule)

        st.write("**Best Fitness Value:**", best_fitness)

        # Show divider after button is pressed
        st.markdown("---")

    st.markdown("Developed with Streamlit and Python")
