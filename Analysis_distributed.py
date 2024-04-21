from distributed import DistributedPlanningSolver
from run_experiments import import_mapf_instance
import concurrent.futures
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from math import *
import os
import scipy.stats as stats
import pandas as pd
import pingouin as pg
from scipy.stats import spearmanr

class Analysis:
    def __init__(self, input_path, timeout_time, threshold_percent):
        self.input_path = input_path # e.g. 'instances\\map1.txt'
        self.timeout_time = timeout_time # amount of time to wait before calling an infinite loop when finding a solution
        self.output_path = 'instances\\Output_generator.txt'
        self.threshold_percent = threshold_percent


    def f(self, solver): # needed for find_solution()
        return solver.find_solution()

    def run_with_timeout(self, func, args, timeout, seed_number): # needed for find_solution()
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

    def has_converged(self, values, window_size, min_consecutive_windows): #needed for compare_performance_methods()
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
                if relative_change <= self.threshold_percent / 100.0:
                    consecutive_below_threshold += 1
                else:
                    consecutive_below_threshold = 0
            if consecutive_below_threshold >= min_consecutive_windows:
                return True
        return False

    def top_bottom_generate_agents_on_map(self, num_agents, seed_val):
        random.seed(seed_val)

        try:
            with open(self.input_path, 'r') as input_file:
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
                with open(self.output_path, 'w') as output_file:
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


    def random_generate_agents_on_map(self, num_agents, seed_val):
        random.seed(seed_val)

        try:
            with open(self.input_path, 'r') as input_file:
                lines = input_file.readlines()

                num_rows, num_columns = map(int, lines[0].strip().split())

                map_content = [[cell for cell in line.strip() if cell in ".@"] for line in lines[1:1 + num_rows]]

                blocked_positions = set((x, y) for y, row in enumerate(map_content) for x, cell in enumerate(row) if cell == "@")

                available_positions = [(x, y) for x in range(num_columns) for y in range(num_rows) if (x, y) not in blocked_positions]

                random.shuffle(available_positions)

                #Adjust output file from here
                with open(self.output_path, 'w') as output_file:
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


    def left_right_generate_agents_on_map(self, num_agents, seed_val):
        random.seed(seed_val)

        try:
            with open(self.input_path, 'r') as input_file:
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
                with open(self.output_path, 'w') as output_file:
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


    def find_solutions(self, agent_generator, num_agents, amount_of_simulations, method, add_on, steps_ahead, scope_rad,\
                       comparison=False, heat_map=False):

        analysis = {'success rate': 0, 'system performance per sim': []}

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
                    self.random_generate_agents_on_map(num_agents, seed_number)
                elif agent_generator == 'left-right':
                    self.left_right_generate_agents_on_map(num_agents, seed_number)
                elif agent_generator == 'top-bottom':
                    self.top_bottom_generate_agents_on_map(num_agents, seed_number)
                my_map, starts, goals = import_mapf_instance(self.output_path)
                solver = DistributedPlanningSolver(my_map, starts, goals, method, add_on, steps_ahead, scope_rad)
                paths = self.run_with_timeout(self.f, [solver], self.timeout_time, seed_number)
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

    def success_result(self, agent_generator, method, add_on, max_agents, amount_of_simulations, steps_ahead, scope_rad):
        success = []
        num_agents_range = range(1,max_agents+1)
        for num_agents in num_agents_range:
            print("current number of agents = ", num_agents)
            analysis = self.find_solutions(agent_generator = agent_generator, \
                                  num_agents=num_agents, amount_of_simulations=amount_of_simulations, \
                                           method=method, add_on=add_on, steps_ahead=steps_ahead, scope_rad=scope_rad)
            success.append(analysis['success rate'])

        return success

    def success_plotter(self, agent_generator, max_agents, amount_of_simulations, steps_ahead, scope_rad):
        success_output = {'Explicit': None, 'Random': None, 'Implicit': None}
        for method in ['Explicit', 'Random', 'Implicit']:
            print("This method's success rate is now being evaluated: ", method)
            add_on = False
            if 'add-on' in method:
                add_on = True
                method_name = method.replace(" with add-on", "")
            else:
                method_name = method
            success = self.success_result(agent_generator=agent_generator, method=method_name, add_on=add_on,
                                                    max_agents=max_agents, amount_of_simulations=amount_of_simulations, steps_ahead=steps_ahead,
                                                    scope_rad=scope_rad)
            success_output[method] = success

        # Setting up the plot
        print("DATA: ", success_output)

        print("PLOTTING SUCCESS RATE FIGURE NOW")
        plt.figure(figsize=(10, 6))

        # Plot each method's results with a distinct line
        num_agents_range = range(1, max_agents + 1)
        for method, successes in success_output.items():
            plt.plot(num_agents_range, successes, marker='o', linestyle='-', label=method)

        plt.xlabel('Number of Agents')
        plt.ylabel('Success Rate')
        plt.title('Success Rate vs. Number of Agents')
        plt.grid(True)
        plt.legend()  # Include a legend to differentiate the lines
        print("SAVING SUCCESS RATE GRAPH")
        plt.savefig(os.path.join('Graphs', 'success_rate_figure.png'))
        print("CLOSE GRAPH TO OBTAIN DATA")
        plt.show()
        return


    def compare_performance_methods(self, agent_generator, num_agents, performance_indicator, methods2compare, add_on, \
                                    steps_ahead, scope_rad, plotting=True):

        result = dict()
        for method in methods2compare:
            cv_values = []
            cv_has_converged = False
            amount_of_simulations = 1
            print("CURRENT METHOD = ", method)
            result_method = []
            while cv_has_converged is False or amount_of_simulations < 10: #minimum of X simulations tried
                analysis = self.find_solutions(agent_generator=agent_generator, num_agents=num_agents, \
                                               amount_of_simulations=amount_of_simulations, method=method, add_on=add_on,
                                               steps_ahead=steps_ahead, scope_rad=scope_rad, comparison=True)

                # with comparison=True only one simulation is done everytime in this loop, saving time when running this while loop
                for system_performance in analysis['system performance per sim']:
                    if system_performance is not None: #do not take performance into account of a failed simulation
                        result_method.append(system_performance[performance_indicator])

                        std_dev = np.std(result_method)
                        mean = np.mean(result_method)
                        cv = std_dev / mean if mean != 0 else float('inf')
                        cv_values.append(cv)
                        cv_has_converged = self.has_converged(values=cv_values, window_size=3, \
                                                         min_consecutive_windows=2) #set convergence determination parameters here

                if amount_of_simulations == 30: #after X simulations it stops evaluating the performance for this method
                    print("COEFFICIENT OF VARIATION NOT ABLE TO CONVERGE FOR FOLLOWING METHOD: " + method)
                    break
                amount_of_simulations += 1
            result[method] = [result_method, cv_values]



        print("CALCULATING SIGNIFICANCE...")

        significance = dict()
        for method1, group1 in result.items():
            for method2, group2 in result.items():
                if method1 != method2 and (method1, method2) not in significance.keys() and (method2, method1) not in significance.keys():
                    t_stat, p_val = stats.ttest_ind(group1[0], group2[0])
                    significance[(method1, method2)] = t_stat, p_val

        if plotting == True:
            print("PLOTTING THE RESULTS NOW...")
            labels, data, cv_values = [], [], []
            for key, values in result.items():
                labels.append(key)
                data.append(values[0])
                cv_values.append(values[1])

            # Calculate the number of rows for the coefficients of variation plots
            num_rows = len(methods2compare)

            # Create a figure with subplots for the coefficients of variation
            fig, axs = plt.subplots(num_rows, 1, figsize=(10, 4 * num_rows))

            # Plot coefficients of variation for each method
            for i, method in enumerate(methods2compare):
                axs[i].plot(range(1, len(cv_values[i]) + 1), cv_values[i], 'tab:blue')
                axs[i].set_title(f'Coefficient of Variation - {method} Method')
                axs[i].set_xlabel('Number of Simulations')
                axs[i].set_ylabel('CV')

            # Adjust the spacing between the subplots
            plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for adequate spacing

            # Show the plots for coefficients of variation
            plt.show()

            # Now, create a separate figure for the boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=data)
            new_labels = [f"{label} (n={len(values[0])})" for label, values in result.items()]
            plt.xticks(range(len(labels)), new_labels)
            plt.title("Boxplot of Distributed Method Performance: " + performance_indicator)
            plt.xlabel("Method")
            plt.ylabel("Performance")

            # Create the legend text
            legend_text = '\n'.join([f"{pair[0]} vs {pair[1]}: t={t_vals[0]:.2f}, p={t_vals[1]:.3f}"
                                     for pair, t_vals in significance.items()])
            plt.figtext(0.95, 0.95, legend_text, ha="right", va="top", fontsize=9,
                        bbox={"boxstyle": "round", "facecolor": "white"})

            # Show the boxplot
            plt.show()


            return result, significance

        else:
            return result, significance


    def compare_performance_extension(self, agent_generator, num_agents, performance_indicator, methods2compare, steps_ahead, scope_rad):

        result, significance = self.compare_performance_methods(agent_generator, num_agents, performance_indicator, \
                                      methods2compare, add_on=False, steps_ahead=steps_ahead, scope_rad=scope_rad, plotting=False)
        result_with_extension, significance_with_extension = self.compare_performance_methods(agent_generator, num_agents, \
                       performance_indicator, methods2compare, add_on=True, steps_ahead=steps_ahead, scope_rad=scope_rad, plotting=False)

        print("CALCULATING SIGNIFICANCE WITH ADD ON...")
        significance = dict()
        for method, data in result.items():
            t_stat, p_val = stats.ttest_ind(result[method][0], result_with_extension[method][0])
            method_added = method + " extended"
            significance[(method, method_added)] = t_stat, p_val

        print("PLOTTING THE RESULTS WITH ADD ON NOW...")
        labels, data = [], []
        for key, values in result.items():
            labels.append(key)
            data.append(values[0])

        for key, values in result_with_extension.items():
            label = key + " extended"
            labels.append(label)
            data.append(values[0])

        result_total = dict()
        for i in range(len(labels)):
            result_total[labels[i]] = data[i]

        # Now, create a separate figure for the boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data)
        new_labels = [f"{label} (n={len(data)})" for label, data in result_total.items()]
        plt.xticks(range(len(labels)), new_labels)
        plt.title("Boxplot of Distributed Method Performance: " + performance_indicator)
        plt.xlabel("Method")
        plt.ylabel("Performance")

        # Create the legend text
        legend_text = '\n'.join([f"{pair[0]} vs {pair[1]}: t={t_vals[0]:.2f}, p={t_vals[1]:.3f}"
                                 for pair, t_vals in significance.items()])
        plt.figtext(0.95, 0.95, legend_text, ha="right", va="top", fontsize=9,
                    bbox={"boxstyle": "round", "facecolor": "white"})

        # Show the boxplot
        plt.show()

        return result, result_with_extension, significance


    def create_heat_map(self, agent_generator, num_agents, method, title_heat_map, waiting_cells, \
                        add_on, steps_ahead, scope_rad, amount_of_simulations):

        analysis, my_map = self.find_solutions(agent_generator=agent_generator, num_agents=num_agents, \
                                               amount_of_simulations=amount_of_simulations, add_on=add_on,
                                               steps_ahead=steps_ahead,
                                               scope_rad=scope_rad, method=method, heat_map=True)
        heat_map = [[np.nan if cell else 0 for cell in row] for row in my_map]
        for perf_dict in analysis['system performance per sim']:
            if perf_dict is not None:
                paths = perf_dict['agent paths with waiting']
                for path in paths:
                    if waiting_cells == False:  # if we want to see where every agent passes
                        for cell in path:
                            heat_map[cell[0]][cell[1]] += (
                                        1 / amount_of_simulations)  # normalize with amount of simulations
                    else:
                        cell_prev = 0
                        for cell in path:
                            if cell == path[-1]:
                                break
                            if cell == cell_prev and cell != path[
                                0]:  # if agent has to wait, then the heat map gets hotter
                                heat_map[cell[0]][cell[1]] += (
                                            1 / amount_of_simulations)  # normalize with amount of simulations
                            cell_prev = cell

        # Create the heatmap
        plt.figure(figsize=(12, 8))
        print("creating heat map...")
        sns.heatmap(heat_map, cmap="coolwarm", annot=True, cbar=True, linewidths=.5,
                    cbar_kws={'label': 'Average amount of agents passing per simulation'})
        print("adding title...")
        plt.title("Heat Map Visualization")
        print("saving heat map...")
        filename = os.path.join('Graphs', title_heat_map)
        plt.savefig(filename)
        plt.close()  # Close the figure after saving
        print(f"Graph saved as {filename}")
        print('Stop manually!')

        return heat_map


    def local_sensitivity_analysis(self, agent_generator, num_agents, performance_indicators, methods2compare, add_on, \
                                    steps_ahead, scope_rad, dP, parameters,map):
        steps_ahead_min = int(floor(steps_ahead - dP * steps_ahead))
        steps_ahead_plus = int(ceil(steps_ahead + dP * steps_ahead))

        scope_rad_min = int(floor(scope_rad - dP * scope_rad))
        scope_rad_plus = int(ceil(scope_rad + dP * scope_rad))

        num_agents_min = int(floor(num_agents - dP * num_agents))
        num_agents_plus = int(ceil(num_agents + dP * num_agents))

        results = dict()
        for performance_indicator in performance_indicators:
            results[performance_indicator] = dict()
            for parameter in parameters:
                if parameter == 'Scope':
                    result_scope_min, significance_scope_min = self.compare_performance_methods(agent_generator, num_agents, performance_indicator, \
                                          methods2compare, add_on, steps_ahead, scope_rad_min, plotting=False)

                    result_scope, significance_scope = self.compare_performance_methods(agent_generator, num_agents, performance_indicator, \
                                                   methods2compare, add_on, steps_ahead, scope_rad, plotting=False)

                    result_scope_plus, significance_scope_plus = self.compare_performance_methods(agent_generator, num_agents, \
                           performance_indicator, methods2compare, add_on, steps_ahead, scope_rad_plus, plotting=False)

                    results[performance_indicator]['Scope'] = [[result_scope_min, significance_scope_min], [result_scope, significance_scope], \
                                        [result_scope_plus, significance_scope_plus]]

                if parameter == 'Agents':
                    result_scope_min, significance_scope_min = self.compare_performance_methods(agent_generator, num_agents_min, performance_indicator,
                                            methods2compare, add_on, steps_ahead, scope_rad_min, plotting=False)

                    result_scope, significance_scope = self.compare_performance_methods(agent_generator, num_agents, performance_indicator, \
                                                       methods2compare, add_on, steps_ahead, scope_rad, plotting=False)

                    result_scope_plus, significance_scope_plus = self.compare_performance_methods(agent_generator, num_agents_plus, performance_indicator,
                                                               methods2compare, add_on, steps_ahead, scope_rad_plus, plotting=False)

                    results[performance_indicator]['Agents'] = [[result_scope_min, significance_scope_min], [result_scope, significance_scope], \
                                        [result_scope_plus, significance_scope_plus]]

                if parameter == 'Steps ahead':
                    result_steps_min, significance_steps_min = self.compare_performance_methods(agent_generator, num_agents,
                                        performance_indicator, methods2compare, add_on, steps_ahead_min, scope_rad_min, plotting=False)

                    result_steps, significance_steps = self.compare_performance_methods(agent_generator, num_agents,
                                                    performance_indicator, methods2compare, add_on, steps_ahead, scope_rad, plotting=False)

                    result_steps_plus, significance_steps_plus = self.compare_performance_methods(agent_generator, num_agents,
                                        performance_indicator, methods2compare, add_on, steps_ahead_plus, scope_rad_plus, plotting=False)

                    results[performance_indicator]['Steps ahead'] = [[result_steps_min, significance_steps_min], [result_steps, significance_steps], \
                                              [result_steps_plus, significance_steps_plus]]


        print("PERFORMING LOCAL SENSITIVITY ANALYSIS...")
        sensitivity = dict()
        for performance, sense in results.items():
            sensitivity[performance] = dict()
            for parameter, result in sense.items():
                sensitivity[performance][parameter] = dict()
                for method in methods2compare:
                    sensitivity[performance][parameter][method] = []

                order = [1, 0, 2]
                for index in order:
                    for method, data in result[index][0].items():
                        if index == 1:
                            sensitivity[performance][parameter][method].append(np.mean(data[0]))
                        if index == 0:
                            mean_min = np.mean(data[0])
                            mean = sensitivity[performance][parameter][method][0]
                            S_min = (mean - mean_min) / dP
                            sensitivity[performance][parameter][method].append(S_min)
                        if index == 2:
                            mean_plus = np.mean(data[0])
                            mean = sensitivity[performance][parameter][method][0]
                            S_plus = (mean_plus - mean) / dP
                            sensitivity[performance][parameter][method].append(S_plus)

        print('For dP = ', dP, ' sensitivity dictionary is ', sensitivity)

        print("DUMPING LOCAL SENSITIVITY DATA...")

        extracted_data = []
        for category, subcategories in sensitivity.items():
            for subcategory, methods in subcategories.items():
                for method, values in methods.items():
                    # Only select the second (S^-) and third (S^+) values
                    s_minus, s_plus = values[1:3]
                    extracted_data.append({
                        'Performance indicator': category,
                        'Parameter': subcategory,
                        'Method': method,
                        'S-': s_minus,
                        'S+': s_plus
                    })

        df = pd.DataFrame(extracted_data)
        df.to_excel(f'Local_sensitivity_3{parameter,map}.xlsx', engine='openpyxl', index=False)
        print("LOCAL SENSITIVITY DATA SAVED IN Local_sensitivity.xlsx...")

        return df

    def global_sensitivity(self, parameters, ranges, n_samples=5):
        # Define base scenario
        agent_generator = 'left-right'
        num_agents = 5
        method = 'Implicit'
        add_on = False
        steps_ahead = 10
        scope_rad = 2
        performance_indicator = 'average travel time'
        amount_of_simulations = 15

        # Generate Latin Hypercube Samples for the two parameters
        lhs_1 = np.random.uniform(ranges[0][0], ranges[0][1]+1, n_samples)
        lhs_2 = np.random.uniform(ranges[1][0], ranges[1][1]+1, n_samples)

        # Create a dataframe to store the parameter combinations: Sampling
        par_1 = np.arange(ranges[0][0], ranges[0][1]+1, 1)
        par_2 = np.arange(ranges[1][0], ranges[1][1]+1, 1)

        param_df = pd.DataFrame(columns=[parameters[0], parameters[1]])
        for i in par_1:
            for j in par_2:
                combination = pd.DataFrame([{parameters[0]: i, parameters[1]: j}])
                param_df = pd.concat([param_df, pd.DataFrame(combination)], ignore_index=True)

        # Initialize lists to store the results
        correlation_coefficients = []
        p_values = []
        partial_correlation_coefficients = []

        data = []

        df = pd.DataFrame()

        # Loop through each parameter combination
        for i, row in param_df.iterrows():
            print(f'COMBINATION {i} STARTING')
            # Set the parameters to the current combination
            self.param1 = row[parameters[0]]
            self.param2 = row[parameters[1]]


            # Run simulation
            if str(parameters[0]) == 'num_agents':
                num_agents = self.param1
            elif str(parameters[1]) == 'num_agents':
                num_agents = self.param2

            if str(parameters[0]) == 'scope_rad':
                scope_rad = self.param1
            elif str(parameters[1]) == 'scope_rad':
                scope_rad = self.param2

            if str(parameters[0]) == 'steps_ahead':
                steps_ahead = self.param1
            elif str(parameters[1]) == 'steps_ahead':
                steps_ahead = self.param2

            cv_values = []
            cv_has_converged = False
            amount_of_simulations = 1
            result_method = []
            success = 0
            while cv_has_converged is False or amount_of_simulations < 10:  # minimum of X simulations tried
                analysis = self.find_solutions(agent_generator=agent_generator, num_agents=num_agents, \
                                               amount_of_simulations=amount_of_simulations, method=method,
                                               add_on=add_on,
                                               steps_ahead=steps_ahead, scope_rad=scope_rad, comparison=True)
                success += analysis['success rate']

                # with comparison=True only one simulation is done everytime in this loop, saving time when running this while loop

                for system_performance in analysis['system performance per sim']:
                    if system_performance is not None:  # do not take performance into account of a failed simulation
                        result_method.append(system_performance[performance_indicator])

                        std_dev = np.std(result_method)
                        mean = np.mean(result_method)
                        cv = std_dev / mean if mean != 0 else float('inf')
                        cv_values.append(cv)
                        cv_has_converged = self.has_converged(values=cv_values, window_size=3, \
                                                              min_consecutive_windows=2)  # set convergence determination parameters here

                if amount_of_simulations == 10:  # after X simulations it stops evaluating the performance for this method
                    print("COEFFICIENT OF VARIATION NOT ABLE TO CONVERGE FOR FOLLOWING METHOD: " + method)
                    break
                amount_of_simulations += 1

            if analysis['success rate'] == 0.0 or cv_has_converged == False:
                param_df.drop(i, inplace=True)
                continue

            if analysis['system performance per sim'] is not None:
                system_performance = analysis['system performance per sim'][0]
                att = {'average travel time': system_performance['average travel time']}
                performance = pd.DataFrame([att])
                performance[str(parameters[0])] = self.param1
                performance[str(parameters[1])] = self.param2
                df = pd.concat([df, performance], ignore_index=True)

        results = df

        CV = pd.DataFrame(cv_values)

        for output_var in results.columns:
            print('OUTPUT', output_var)
            if output_var != parameters[0] and output_var != parameters[1]:
                # Calculate Spearman correlation coefficient and p-value
                corr_coef, p_val = spearmanr(param_df.values, results[output_var].values)
                correlation_coefficients.append(corr_coef)
                p_values.append(p_val)
                print('CC', correlation_coefficients)
                print('P-VALUE', p_values)

                # Calculate Partial correlation coefficient
                partial_corr = pg.partial_corr(data=results, x=str(parameters[0]), y=str(parameters[1]), covar='average travel time', method='spearman')['r']
                partial_correlation_coefficients.append(partial_corr)
                print('PCC', partial_correlation_coefficients)
            else:
                correlation_coefficients.append(None)
                p_values.append(None)
                partial_correlation_coefficients.append(None)

        # Create a dataframe to store sensitivity results
        sensitivity_results = pd.DataFrame({
            'Output Variable': results.columns,
            'Correlation Coefficient': correlation_coefficients,
            'P-value': p_values,
            'Partial Correlation Coefficient': partial_correlation_coefficients
        })
        # sensitivity_results = []
        sensitivity_results.to_excel(f'instances/globalsens_IMPL_{parameters[0]}_{parameters[1]}.xlsx', index=False)
        results.to_excel(f'instances/global_IMPL_{parameters[0]}_{parameters[1]}.xlsx', index=False)
        print('DONE')
        return sensitivity_results, results


map1_analysis = Analysis(input_path='instances\\map1.txt', timeout_time=2, threshold_percent=30)
map2_analysis = Analysis(input_path='instances\\map2.txt', timeout_time=2, threshold_percent=30)
map3_analysis = Analysis(input_path='instances\\map3.txt', timeout_time=2, threshold_percent=30)

'''
OPTIONS FOR PERFORMANCE INDICATORS: ['maximum time', 'total time', 'total distance traveled', 'total amount of conflicts', 'average travel time',
'average travel distance', 'average conflicts']

OPTIONS FOR SENSITIVITY PARAMETERS: ['Scope', 'Agents', 'Steps ahead']

'''

lst_indicators = ['maximum time', 'total time', 'total distance traveled', 'total amount of conflicts', 'average travel time', 'average travel distance', 'average conflicts']
lst_parameters = ['Scope', 'Agents', 'Steps ahead']
lst_maps_analysis = [map1_analysis, map2_analysis, map3_analysis]