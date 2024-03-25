from distributed import DistributedPlanningSolver
from run_experiments import import_mapf_instance
import concurrent.futures
import random
import matplotlib.pyplot as plt
import os

import numpy as np
import seaborn as sns

import plotly.graph_objects as go
import os



def top_bottom_generate_agents_on_map(input_path, output_path, num_agents, seed_val):
    random.seed(seed_val)

    try:
        with open(input_path, 'r') as input_file:
            lines = input_file.readlines()

            num_rows, num_columns = map(int, lines[0].strip().split())

            map_content = [[cell for cell in line.strip() if cell in ".@"] for line in lines[1:1 + num_rows]]

            blocked_positions = set((x, y) for y, row in enumerate(map_content) for x, cell in enumerate(row) if cell == "@")

            # Define specific rows for start/end locations
            top_rows = [1, 2]
            bottom_rows = [num_rows - 2, num_rows - 3]

            # Filter out positions based on rows and blocked positions
            top_positions = [(x, y) for y in top_rows for x in range(num_columns) if (x, y) not in blocked_positions]
            bottom_positions = [(x, y) for y in bottom_rows for x in range(num_columns) if (x, y) not in blocked_positions]

            # Check if there are enough positions for half the agents on each side
            half_agents = num_agents // 2
            if len(top_positions) < half_agents or len(bottom_positions) < half_agents:
                print("Not enough available positions for agents. Adjusting number of agents.")
                half_agents = min(len(top_positions), len(bottom_positions))
                num_agents = 2 * half_agents

            random.shuffle(top_positions)
            random.shuffle(bottom_positions)

            #Adjust output file from here
            with open(output_path, 'w') as output_file:
                output_file.write(f"{num_rows} {num_columns}\n")

                for line in map_content:
                    output_file.write(" ".join(line) + "\n")

                output_file.write(f"{num_agents}\n")

                for agent in range(num_agents):
                    if agent < half_agents:
                        # Agents starting in top rows
                        start_x, start_y = top_positions.pop()
                        goal_x, goal_y = bottom_positions.pop()
                    else:
                        # Agents starting in bottom rows
                        start_x, start_y = bottom_positions.pop()
                        goal_x, goal_y = top_positions.pop()

                    output_file.write(f"{start_y} {start_x} {goal_y} {goal_x}\n")

                output_file.write(f"\n \n \n seed = {seed_val}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def random_generate_agents_on_map(input_path, output_path, num_agents, seed_val):
    random.seed(seed_val)

    try:
        with open(input_path, 'r') as input_file:
            lines = input_file.readlines()

            num_rows, num_columns = map(int, lines[0].strip().split())

            map_content = [[cell for cell in line.strip() if cell in ".@"] for line in lines[1:1 + num_rows]]

            blocked_positions = set((x, y) for y, row in enumerate(map_content) for x, cell in enumerate(row) if cell == "@")

            available_positions = [(x, y) for x in range(num_columns) for y in range(num_rows) if (x, y) not in blocked_positions]

            random.shuffle(available_positions)

            #Adjust output file from here
            with open(output_path, 'w') as output_file:
                output_file.write(f"{num_rows} {num_columns}\n")

                for line in map_content:
                    output_file.write(" ".join(line) + "\n")

                output_file.write(f"{num_agents}\n")

                for agent in range(1, num_agents + 1):
                    start_x, start_y = available_positions.pop()
                    goal_x, goal_y = available_positions.pop()
                    output_file.write(f"{start_y} {start_x} {goal_y} {goal_x}\n")

                output_file.write(f"\n \n \n seed = {seed_val}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def left_right_generate_agents_on_map(input_path, output_path, num_agents, seed_val):
    random.seed(seed_val)

    try:
        with open(input_path, 'r') as input_file:
            lines = input_file.readlines()

            num_rows, num_columns = map(int, lines[0].strip().split())

            map_content = [[cell for cell in line.strip() if cell in ".@"] for line in lines[1:1 + num_rows]]

            blocked_positions = set(
                (x, y) for y, row in enumerate(map_content) for x, cell in enumerate(row) if cell == "@")

            # Define specific columns for start/end locations
            left_columns = [1, 2]
            right_columns = [21, 22]

            # Filter out positions based on columns and blocked positions
            left_positions = [(x, y) for x in left_columns for y in range(num_rows) if
                              (x, y) not in blocked_positions]
            right_positions = [(x, y) for x in right_columns for y in range(num_rows) if
                               (x, y) not in blocked_positions]

            # Check if there are enough positions for half the agents on each side
            half_agents = num_agents // 2
            if len(left_positions) < half_agents or len(right_positions) < half_agents:
                print("Not enough available positions for agents. Adjusting number of agents.")
                half_agents = min(len(left_positions), len(right_positions))
                num_agents = 2 * half_agents

            random.shuffle(left_positions)
            random.shuffle(right_positions)

            #Adjust output file from here
            with open(output_path, 'w') as output_file:
                output_file.write(f"{num_rows} {num_columns}\n")

                for line in map_content:
                    output_file.write(" ".join(line) + "\n")

                output_file.write(f"{num_agents}\n")

                for agent in range(num_agents):
                    if agent < half_agents:
                        # Agents traveling from left to right
                        start_x, start_y = left_positions.pop()
                        goal_x, goal_y = right_positions.pop()
                    else:
                        # Agents traveling from right to left
                        start_x, start_y = right_positions.pop()
                        goal_x, goal_y = left_positions.pop()

                    output_file.write(f"{start_y} {start_x} {goal_y} {goal_x}\n")

                output_file.write(f"\n \n \n seed = {seed_val}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def f(solver):
    return solver.find_solution()

def run_with_timeout(func, args, timeout, seed_number):
    global inf_loop
    executor = concurrent.futures.ThreadPoolExecutor()

    try:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print(seed_number, "has an infinite loop")
            inf_loop += 1
        except Exception as e:
            print(f"An error occurred in the thread: {e}")
    finally:
        # Ensure the executor is properly shut down
        executor.shutdown(wait=False)

    # Return None to indicate failure or timeout
    return None

def find_solutions(agent_generator, timeout_time, input_file_path, num_agents, amount_of_simulations, method, comparison=False, heat_map=False):

    analysis = {'success rate': 0, 'system performance per sim': []}
    output_file_path = 'instances\\Output_generator.txt'

    solutions = 0
    inf_loop = 0
    no_solutions = 0

    if comparison == True:
        start_amount = amount_of_simulations
    else:
        start_amount = 1

    for seed_number in range(start_amount, amount_of_simulations + 1):
        print("current seed = ", seed_number)
        analysis['system performance per sim'].append(None)
        try:
            if agent_generator == 'random':
                random_generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number)
            elif agent_generator == 'left-right':
                left_right_generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number)
            elif agent_generator == 'top-bottom':
                top_bottom_generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number)
            my_map, starts, goals = import_mapf_instance(output_file_path)
            solver = DistributedPlanningSolver(my_map, starts, goals, method)
            paths = run_with_timeout(f, [solver], timeout_time, seed_number)
            solver.performance(paths)
            if paths is not None:
                solutions += 1
                print(seed_number, "HAS A SOLUTION!")
                analysis["system performance per sim"][-1] = solver.performance_system #if no solution found, performance dict = 0

        except BaseException as e:  #Exception if no solution is found
            no_solutions += 1
            print(seed_number, "has no solution")

    print('######### RESULTS ##########')
    print(no_solutions, 'times of not finding a solution')
    print(inf_loop, 'times of infinite looping')
    print(solutions, 'solutions')
    success_rate = solutions/(no_solutions + inf_loop + solutions) * 100
    print(success_rate, '% success rate')
    print('######### RESULTS ##########')

    analysis['success rate'] = success_rate

    if heat_map is True:
        return analysis, my_map
    else:
        return analysis

def success_plotter(agent_generator, method, max_agents, input_file_path, amount_of_simulations, title_success_plot):
    success = []
    num_agents_range = range(1,max_agents+1)
    for num_agents in num_agents_range:
        print("current number of agents = ", num_agents)
        analysis = find_solutions(agent_generator = agent_generator, timeout_time=2, input_file_path=input_file_path, \
                              num_agents=num_agents, amount_of_simulations=amount_of_simulations, method=method)
        success.append(analysis['success rate'])

    plt.figure(figsize=(10, 6))
    plt.plot(num_agents_range, success, marker='o', linestyle='-')
    plt.xlabel('Number of Agents')
    plt.ylabel('Success Rate')
    plt.title('Success Rate vs. Number of Agents')
    plt.grid(True)

    filename = os.path.join('Graphs', title_success_plot)
    plt.savefig(filename)
    plt.close()  # Close the figure after saving
    print(f"Graph saved as {filename}")
    plt.show()
    return
#success_plotter(agent_generator='left-right', method='explicit', max_agents=10, input_file_path='instances\\map1.txt', amount_of_simulations=1, \
#                title_success_plot='success_rate_vs_number_of_agents_map1_explicit_leftright.png')




#COMPARE THE PERFORMANCE INDICATORS OF THE DIFFERENT METHODS, FOR A SPECIFIC MAP/NUM_AGENTS/AGENT_GENERATOR
def has_converged(values, threshold_percent, window_size, min_consecutive_windows):
    if len(values) < window_size:
        return False
    averages = [sum(values[i:i + window_size]) / window_size for i in range(len(values) - window_size + 1)]
    consecutive_below_threshold = 0
    for i in range(1, len(averages)):
        if averages[i-1] == 0:
            if averages[i] != 0:
                consecutive_below_threshold = 0
            else:
                consecutive_below_threshold += 1
        else:
            relative_change = abs((averages[i] - averages[i-1]) / averages[i-1])
            if relative_change <= threshold_percent / 100.0:
                consecutive_below_threshold += 1
            else:
                consecutive_below_threshold = 0
        if consecutive_below_threshold >= min_consecutive_windows:
            return True
    return False

def compare_performance_methods(agent_generator, num_agents, input_file_path, performance_indicator, methods2compare):

    result = dict()

    for method in methods2compare:
        cv_values = []
        cv_has_converged = False
        amount_of_simulations = 1
        print("CURRENT METHOD = ", method)
        result_method = []
        while cv_has_converged is False or amount_of_simulations < 5: #minimum of X simulations tried
            analysis = find_solutions(agent_generator=agent_generator, timeout_time=2, input_file_path=input_file_path, \
                                  num_agents=num_agents, amount_of_simulations=amount_of_simulations, method=method, comparison=True)
            #with comparison=True only one simulation is done everytime in this loop, saving time when running this while loop
            for system_performance in analysis['system performance per sim']:
                if system_performance is not None: #do not take performance into account of a failed simulation
                    result_method.append(system_performance[performance_indicator])

                    std_dev = np.std(result_method)
                    mean = np.mean(result_method)
                    cv = std_dev / mean if mean != 0 else float('inf')
                    cv_values.append(cv)
                    cv_has_converged = has_converged(values=cv_values, threshold_percent=30, window_size=3, \
                                                     min_consecutive_windows=2) #set convergence determination parameters here

            if amount_of_simulations == 20: #after X simulations it stops evaluating the performance for this method
                print("COEFFICIENT OF VARIATION NOT ABLE TO CONVERGE FOR FOLLOWING METHOD: " + method)
                break
            amount_of_simulations += 1
        result[method] = result_method

    labels, data = [], []
    for key, values in result.items():
        labels.append(key)
        data.append(values)

    print("PLOTTING THE RESULTS NOW")
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    new_labels = [f"{key} (n={len(values)})" for key, values in result.items()]
    plt.xticks(range(len(labels)), new_labels)
    plt.title("Boxplot of Distributed Method Performance: " + performance_indicator)
    plt.xlabel("Method")
    plt.ylabel("Performance")
    plt.show()

    #Maybe add something to save the figure

    return
'''
OPTIONS FOR PERFORMANCE INDICATORS: 'maximum time', 'total time', 'total distance traveled', 'total amount of conflicts', 'average travel time',
'average travel distance', 'average conflicts'
'''
#compare_performance_methods(agent_generator='left-right', num_agents=7, input_file_path='instances\\map1.txt', \
#                            performance_indicator='average travel time', methods2compare=['Explicit', 'Random'])


def create_heat_map(agent_generator, num_agents, input_file_path, amount_of_simulations, method, title_success_plot, waiting_cells):

    analysis, my_map = find_solutions(agent_generator=agent_generator, timeout_time=2, input_file_path=input_file_path,
                                      num_agents=num_agents, amount_of_simulations=amount_of_simulations, method=method, heat_map=True)

    heat_map = [[np.nan if cell else 0 for cell in row] for row in my_map]

    for perf_dict in analysis['system performance per sim']:
        if perf_dict is not None:
            paths = perf_dict['agent paths with waiting']
            for path in paths:
                if waiting_cells == False: #if we want to see where every agent passes
                    for cell in path:
                        heat_map[cell[0]][cell[1]] += (1/amount_of_simulations) #normalize with amount of simulations to get an understandable result
                else:
                    cell_prev = 0
                    for cell in path:
                        if cell == path[-1]:
                            break
                        if cell == cell_prev and cell != path[0]: #if agent has to wait, then the heat map gets hotter
                            heat_map[cell[0]][cell[1]] += (1/amount_of_simulations) #normalize with amount of simulations to get an understandable result
                        cell_prev = cell

    # Create the heatmap
    plt.figure(figsize=(12, 8))
    print("creating heat map...")
    sns.heatmap(heat_map, cmap="coolwarm", annot=True, cbar=True, linewidths=.5, cbar_kws={'label': 'Average amount of agents passing per simulation'})
    print("adding title...")
    plt.title("Heat Map Visualization")
    print("saving heat map...")

    filename = os.path.join('Graphs', title_success_plot)
    plt.savefig(filename)
    plt.close()  # Close the figure after saving
    print(f"Graph saved as {filename}")
    print('Stop manually!')

    return heat_map

create_heat_map(agent_generator='top-bottom', num_agents=10, input_file_path='instances\\map1.txt', amount_of_simulations=30, \
                method='Random', title_success_plot='heat_map_random_top_bottom.png', waiting_cells=True)

create_heat_map(agent_generator='top-bottom', num_agents=10, input_file_path='instances\\map1.txt', amount_of_simulations=30, \
                method='Explicit', title_success_plot='heat_map_explicit_top_bottom.png', waiting_cells=True)



