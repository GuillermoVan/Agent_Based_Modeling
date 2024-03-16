import random
from distributed import DistributedPlanningSolver
from run_experiments import import_mapf_instance
import concurrent.futures
import time

def generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number, working_check=False):
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

input_file_path = 'instances\\map1.txt' #change here the type of map you want
output_file_path = 'instances\\Output_generator.txt'
num_agents = 17 #pick the amount of agents that are placed randomly over the map
amount_of_simulations = 5

solutions = 0
inf_loop = 0
no_solutions = 0
timeout_time = 1.5 #put here the amount of time you want to wait before going to next simulation

for seed_number in range(1, amount_of_simulations + 1):
    print("current seed = ", seed_number)
    try:
        generate_agents_on_map(input_file_path, output_file_path, num_agents, seed_number)
        my_map, starts, goals = import_mapf_instance(output_file_path)
        solver = DistributedPlanningSolver(my_map, starts, goals)

        paths = run_with_timeout(f, [solver], timeout_time, seed_number)
        if paths is not None:
            solutions += 1
            print(seed_number, "HAS A SOLUTION!")
    except BaseException as e:  #Exception if no solution is found
        no_solutions += 1
        print(seed_number, "has no solution")

print(no_solutions, 'times of not finding a solution')
print(inf_loop, 'times of infinite looping')
print(solutions, 'solutions')

