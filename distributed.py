"""
This file contains a placeholder for the DistributedPlanningSolver class that can be used to implement distributed planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""

import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost
from distributed_agent_class import DistributedAgent
import numpy as np
from cbs import detect_collision, detect_collisions


class DistributedPlanningSolver(object):
    """A distributed planner"""

    def __init__(self, my_map, starts, goals, method):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        self.CPU_time = 0
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.method = method
        self.num_of_agents = len(goals)
        self.heuristics = []
        self.distributed_agents = []
        self.initial_paths = dict()
        self.performance_agents = dict()
        self.performance_system = dict()
        self.conflict_agents = dict()

        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

    def find_solution(self):
        """
        Finds paths for all agents from start to goal locations.

        Returns:
            result (list): with a path [(s,t), .....] for each agent.
        """
        # Initialize constants
        start_time = timer.time()
        self.CPU_time = timer.time() - start_time

        # Create agent objects with DistributedAgent class and initiate their independent paths
        paths = []
        for i in range(self.num_of_agents):
            newAgent = DistributedAgent(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i)
            self.distributed_agents.append(newAgent)
            paths.append(newAgent.path)
            self.initial_paths[newAgent.id] = newAgent.path.copy()
            self.conflict_agents[newAgent.id] = 0


        time = 0
        arrived = []
        constraints = []

        #while not all agents have reached their target
        while len(arrived) < len(self.distributed_agents) + 1:
            change = True
            while change == True:
                change_check = []

                for agent_1 in self.distributed_agents:
                    # if agent_1 not in arrived:
                    if time >= len(agent_1.path):
                        agent_1.path.append(agent_1.path[-1])
                    agent_1.start = agent_1.path[time]

                    scope_map = self.define_scope(paths, time, agent_1, scope_rad=2)
                    agents_in_scope = self.detect_agent_in_scope(agent_1, scope_map, time)
                    for agent_2 in agents_in_scope:
                        old_length_constraints = len(constraints)
                        if self.method == "Random" or self.method == "Explicit":
                            constraints, change_curr = self.conflict(agent_1, agent_2, time, constraints)
                            if len(constraints) != old_length_constraints: #this part is needed for the conflict performance indicators
                                #print("CONFLICT DETECTED BETWEEN AGENT", agent_1.id, "and", agent_2.id)
                                self.conflict_agents[agent_1.id] += 1
                                self.conflict_agents[agent_2.id] += 1
                            change_check.append(change_curr)

                    if agent_1.path[time] == agent_1.goal and agent_1 not in arrived:
                        arrived.append(agent_1)


                if True in change_check:
                    change = True
                    time = 0
                else:
                    change = False
                    time += 1

            if len(arrived) >= len(self.distributed_agents):
                arrived.append('Final check')

        result = []
        for agent in self.distributed_agents:
            result.append(agent.path)

        #GET PERFORMANCE DATA AND VISUALIZE
        self.performance(result)
        self.visualize_performance(self.performance_agents, self.performance_system)

        return result  # Hint: this should be the final result of the distributed planning (visualization is done after planning)

    def define_scope(self, result, timestep, agentID1, scope_rad=2):
        complete_map = [[0 if cell else 1 for cell in row] for row in self.my_map]

        center_row, center_col = agentID1.path[timestep]

        # Create a copy of the complete_map
        scope_map = np.copy(complete_map)

        # Iterate over the cells and set values based on the scope_rad
        for i in range(len(complete_map)):
            for j in range(len(complete_map[i])):
                distance = abs(center_row - i) + abs(center_col - j)  # Manhattan distance
                if distance > scope_rad:
                    scope_map[i][j] = 0

        return scope_map

    def conflict(self, agent_1, agent_2, time, constraints):
        change = False

        for i in range(1,20): # Communicate 3 time steps ahead
            avoidance = False   # Check for avoidance for all 3 time steps
            timestep = time + i  # Timestep which is checked

            while avoidance == False:   # Only continue when collision is avoided
                """ Let agent stay at its final location after path is finished."""
                while timestep + 2 >= len(agent_1.path):
                    agent_1.path.append(agent_1.path[-1])

                while timestep + 2 >= len(agent_2.path):
                    agent_2.path.append(agent_2.path[-1])

                """ Check if collision occurs."""
                if agent_1.path[timestep] == agent_2.path[timestep] or (agent_1.path[timestep] == agent_2.path[timestep-1] and agent_1.path[timestep-1] == agent_2.path[timestep]):

                    """ Create alternative paths."""
                    constraint_temp_1 = []
                    constraint_temp_2 = []

                    for constraint in constraints:
                        if constraint['time'] == time:
                            constraint_temp_1.append(constraint)
                            constraint_temp_2.append(constraint)

                    for j in range(timestep - time + 2):   # Temporary constraints for the other agent's path

                        constraint_temp_1.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_1.id,
                                                'loc': [agent_2.path[j+time]],
                                                'timestep': j,
                                                'time': time})

                        constraint_temp_2.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_2.id,
                                                'loc': [agent_1.path[j+time]],
                                                'timestep': j,
                                                'time': time})

                        constraint_temp_1.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_1.id,
                                                'loc': [agent_2.path[j+time], agent_2.path[j+time-1]],
                                                'timestep': j,
                                                'time': time})

                        constraint_temp_2.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_2.id,
                                                'loc': [agent_1.path[j+time], agent_1.path[j+time-1]],
                                                'timestep': j,
                                                'time': time})

                    path_1 = agent_1.find_solution(constraints=constraint_temp_1)
                    path_2 = agent_2.find_solution(constraints=constraint_temp_2)
                    if self.method == 'Random':
                        agent_1.random = np.random.normal(0,1)
                        agent_2.random = np.random.normal(0,1)

                        while agent_2.random == agent_1.random:
                            agent_2.random = np.random.normal(0,1)

                        if agent_1.random > agent_2.random:
                            agent_2.path = agent_2.path[:time] + path_2
                            for constraint in constraint_temp_2:
                                if constraint not in constraints:
                                    constraints.append(constraint)
                            change = True
                        else:
                            agent_1.path = agent_1.path[:time] + path_1
                            for constraint in constraint_temp_1:
                                if constraint not in constraints:
                                    constraints.append(constraint)
                            change = True
                    elif self.method == 'Explicit':
                        if len(path_1) >= len(path_2): # Agent with longest detour receives priority
                            agent_2.path = agent_2.path[:time] + path_2
                            for constraint in constraint_temp_2:
                                if constraint not in constraints:
                                    constraints.append(constraint)
                            change = True

                        else:
                            agent_1.path = agent_1.path[:time] + path_1
                            for constraint in constraint_temp_1:
                                if constraint not in constraints:
                                    constraints.append(constraint)
                            change = True

                else:   # No collision occurs at timestep
                    avoidance = True

        return constraints, change

    def detect_agent_in_scope(self, checking_agent, map, time):
        detected_agents = []

        for agent in self.distributed_agents:
            if time >= len(agent.path):
                curr_loc = agent.path[-1]
            else:
                curr_loc = agent.path[time]

            if map[curr_loc[0]][curr_loc[1]] == 1 and agent.id != checking_agent.id:
                detected_agents.append(agent)

        return detected_agents

    def performance(self, result):
        #AGENT-SPECIFIC INDICATORS
        distances = dict()
        travel_times = dict()

        for agent in self.distributed_agents:
            performance_per_agent = dict()

            #Determine the agent's path without duplicates in the end and also one without waiting time included
            agent_path = result[agent.id]
            if agent_path[-1] == result[agent.id][-2]:
                for i in range(len(agent_path)-1, 0, -1):
                    if agent_path[i] != agent_path[i - 1]:
                        break
                agent_path = agent_path[:i+1]
            travel_times[agent.id] = len(agent_path) - 1 #fill in travel_times dictionary needed for system wide performance indicators

            pops = []
            for i in range(len(agent_path)-1):
                if agent_path[i] == agent_path[i+1]:
                    pops.append(i)
            agent_path_no_waiting = [location for idx, location in enumerate(agent_path) if idx not in pops]
            distances[agent.id] = len(agent_path_no_waiting) - 1 #fill in distances dictionary needed for system wide performance indicators

            #Fill in the performance indicator dictionary per agent
            performance_per_agent['travel distance / shortest distance'] = (len(agent_path_no_waiting) - 1) / (len(set(self.initial_paths[agent.id])) - 1)
            performance_per_agent['travel time / shortest time'] =  travel_times[agent.id] / (len(self.initial_paths[agent.id]) - 1)
            performance_per_agent['#conflicts / travel time'] =  self.conflict_agents[agent.id] / travel_times[agent.id]
            self.performance_agents[agent.id] = performance_per_agent

        #SYSTEM-WIDE INDICATORS
        self.performance_system['maximum time'] = max([value for key, value in travel_times.items()]) #this is the time in which all agents have reached their destination
        self.performance_system['total time'] = 0 #this is the sum of all times of agents
        for agent, time in travel_times.items():
            self.performance_system['total time'] += time
        self.performance_system['total distance traveled'] = 0
        for agent, distance in distances.items():
            self.performance_system['total distance traveled'] += distance
        self.performance_system['total amount of conflicts'] = sum([value for key, value in self.conflict_agents.items()]) / 2 #divide by 2, because 1 conflict is for 2 agents
        self.performance_system['average travel time'] = sum([value for key, value in travel_times.items()]) / len(result)
        self.performance_system['average travel distance'] = self.performance_system['total distance traveled'] / len(result)
        self.performance_system['average conflicts'] = self.performance_system['total amount of conflicts'] * 2 / len(result) #times two because it is average amount of conflicts per agent

        #CREATE HEAT MAP HERE
        #map_01 = [[0 if cell else 1 for cell in row] for row in self.my_map]
        #for agent_path in result:


        #print(self.conflict_agents)
        #print(self.performance_agents)
        #print(self.performance_system)

    def visualize_performance(self, performance_agents, performance_system):
        # Heat map
        # Graphs
        return None



