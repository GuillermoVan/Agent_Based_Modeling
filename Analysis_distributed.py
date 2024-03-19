import random
from distributed import DistributedPlanningSolver
from run_experiments import import_mapf_instance
import concurrent.futures
import random

'''
def generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number):
    random.seed(seed_number)

    try:
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

            # Grid size
            num_rows, num_columns = map(int, lines[0].strip().split())

            # Map reading
            map_content = [[cell for cell in line.strip() if cell in ".@"]
                           for line in lines[1:1 + num_rows]]

            blocked_positions = set(
                (x, y) for y, row in enumerate(map_content) for x, cell in enumerate(row) if cell == "@")
            available_positions = [(x, y) for x in range(num_columns) for y in range(num_rows) if
                                   (x, y) not in blocked_positions]

            # Checking the feasability of the number of agents
            if len(available_positions) < num_agents:
                print("Not enough available positions for agents. Exiting.")
                return

            # Randomize the available positions
            random.shuffle(available_positions)

            # Writing the output file
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"{num_rows} {num_columns}\n")

                # Writing the map
                for line in map_content:
                    output_file.write(" ".join(line) + "\n")

                output_file.write(f"{num_agents}\n")

                # Writing the agents and their start and goal positions
                for agent_id in range(1, num_agents + 1):
                    start_x, start_y = available_positions.pop()
                    goal_x, goal_y = available_positions.pop()
                    output_file.write(f"{start_y} {start_x} {goal_y} {goal_x}\n")

                output_file.write(f"\n \n \n seed = {seed_number}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
'''

import random

def generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number):
    random.seed(seed_number)

    try:
        with open(input_file_path, 'r') as input_file:
            lines = input_file.readlines()

            # Grid size
            num_rows, num_columns = map(int, lines[0].strip().split())

            # Map reading
            map_content = [[cell for cell in line.strip() if cell in ".@"]
                           for line in lines[1:1 + num_rows]]

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

            # Randomize the positions
            random.shuffle(left_positions)
            random.shuffle(right_positions)

            # Writing the output file
            with open(output_file_path, 'w') as output_file:
                output_file.write(f"{num_rows} {num_columns}\n")

                # Writing the map
                for line in map_content:
                    output_file.write(" ".join(line) + "\n")

                output_file.write(f"{num_agents}\n")

                # Assign start and goal positions
                for agent_id in range(num_agents):
                    if agent_id < half_agents:
                        # Agents traveling from left to right
                        start_x, start_y = left_positions.pop()
                        goal_x, goal_y = right_positions.pop()
                    else:
                        # Agents traveling from right to left
                        start_x, start_y = right_positions.pop()
                        goal_x, goal_y = left_positions.pop()

                    output_file.write(f"{start_y} {start_x} {goal_y} {goal_x}\n")

                output_file.write(f"\n \n \n seed = {seed_number}")

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

def find_solutions(timeout_time, input_file_path, num_agents, amount_of_simulations, method):

    analysis = dict()
    output_file_path = 'instances\\Output_generator.txt'

    solutions = 0
    inf_loop = 0
    no_solutions = 0

    for seed_number in range(1, amount_of_simulations + 1):
        print("current seed = ", seed_number)
        try:
            generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number)
            my_map, starts, goals = import_mapf_instance(output_file_path)
            solver = DistributedPlanningSolver(my_map, starts, goals, method)
            paths = run_with_timeout(f, [solver], timeout_time, seed_number)
            if paths is not None:
                solutions += 1
                print(seed_number, "HAS A SOLUTION!")
        except BaseException as e:  #Exception if no solution is found
            no_solutions += 1
            print(seed_number, "has no solution")

    print('######### RESULTS ##########')
    print(no_solutions, 'times of not finding a solution')
    print(inf_loop, 'times of infinite looping')
    print(solutions, 'solutions')
    print(solutions/(no_solutions + inf_loop + solutions) * 100, '% success rate')
    print('######### RESULTS ##########')

    return analysis

analysis = find_solutions(timeout_time=3, input_file_path='instances\\map1.txt', num_agents=10, amount_of_simulations=50, method="Explicit")
