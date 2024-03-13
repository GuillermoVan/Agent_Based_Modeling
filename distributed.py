"""
This file contains a placeholder for the DistributedPlanningSolver class that can be used to implement distributed planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""

import time as timer
from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost
from distributed_agent_class import DistributedAgent
import numpy as np
from cbs import detect_collision, detect_collisions

#SOPE DEFINITION
#Function 1: input = map, current location -> output = map of 0s and 1s

#DETECT AGENT
#Function 2: input = map of 0s and 1s, current location of all agents -> if agent detected, then output = detected agent's object

#DETECT CONFLICT + COMMUNICATION
#Function 3: input = agent object -> output = True/False conflict detection
#if conflict detected, shortest path without passing conflict point, slowest path 'wins' -> output = path update for agent who 'lost'

#LATER ON WE WILL EXPAND TO MULTIPLE AGENTS IN SAME SCOPE AND SEE HOW THAT WORKS


class DistributedPlanningSolver(object):
    """A distributed planner"""

    def __init__(self, my_map, starts, goals):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        """
        self.CPU_time = 0
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)
        self.heuristics = []
        self.distributed_agents = []
        self.initial_paths = []
        self.performance_agents = dict()
        self.performance_system = dict()

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
            self.initial_paths = paths


        time = 0
        arrived = []
        constraints = []

        #while not all agents have reached their target
        while len(arrived) < len(self.distributed_agents):
            print(arrived)
            for agent_1 in self.distributed_agents:
                if agent_1 not in arrived:
                    if time >= len(agent_1.path):
                        agent_1.path.append(agent_1.path[-1])

                    scope_map = self.define_scope(paths, time, agent_1, scope_rad=2)
                    agents_in_scope = self.detect_agent_in_scope(agent_1, scope_map, time)
                    for agent_2 in agents_in_scope:
                        constraints = self.conflict(agent_1, agent_2, time, constraints)

                    if agent_1.path[time] == agent_1.goal:
                        arrived.append(agent_1)

                    agent_1.start = agent_1.path[time]


            time += 1

        result = []
        for agent in self.distributed_agents:
            result.append(agent.path)

        print(result)

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
        for i in range(1,4): # Communicate 3 time steps ahead
            avoidance = False   # Check for avoidance for all 3 time steps
            timestep = time + i  # Timestep which is checked

            while avoidance == False:   # Only continue when collision is avoided
                """ Let agent stay at its final location after path is finished."""
                while timestep + 1 >= len(agent_1.path):
                    agent_1.path.append(agent_1.path[-1])

                while timestep + 1 >= len(agent_2.path):
                    agent_2.path.append(agent_2.path[-1])

                """ Check if collision occurs."""
                if agent_1.path[timestep] == agent_2.path[timestep]:
                    print(timestep)
                    """ Create alternative paths."""
                    constraint_temp_1 = []
                    constraint_temp_2 = []

                    for j in range(timestep - time + 2):   # Temporary constraints for the other agent's path

                        constraint_temp_1.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_1.id,
                                                'loc': [agent_2.path[j+time]],
                                                'timestep': j})

                        constraint_temp_2.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_2.id,
                                                'loc': [agent_1.path[j+time]],
                                                'timestep': j})

                        constraint_temp_1.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_1.id,
                                                'loc': [agent_2.path[j+time], agent_2.path[j-1+time]],
                                                'timestep': j})

                        constraint_temp_2.append({'positive': False,
                                                'negative': True,
                                                'agent': agent_2.id,
                                                'loc': [agent_1.path[j+time], agent_1.path[j-1+time]],
                                                'timestep': j})

                    path_1 = agent_1.find_solution(constraints=constraint_temp_1)
                    path_2 = agent_2.find_solution(constraints=constraint_temp_2)

                    if len(path_1) >= len(path_2): # Agent with longest detour receives priority
                        agent_2.path = agent_2.path[:time] + path_2
                        constraints.append(constraint_temp_2)
                    else:
                        agent_1.path = agent_1.path[:time] + path_1
                        constraints.append(constraint_temp_1)


                else:   # No collision occurs at timestep
                    avoidance = True

        return constraints

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
        for agent in self.distributed_agents:
            performance_per_agent = dict()
            performance_per_agent['optimal path / traveled distance'] = len(set(self.initial_paths[agent.id])) / len(set(result[agent.id]))
            performance_per_agent['optimal time / travel time'] = len(self.initial_paths[agent.id]) / len(result[agent.id])
            #NEED REST OF CODE TO WORK TO IMPLEMENT THIS LINE BELOW -> conflicts_per_agent needs to be defined
            #performance_per_agent['#conflicts / travel time'] =  conflicts_per_agent / len(result[agent.id])
            self.performance_agents[agent.id] = performance_per_agent

        #SYSTEM-WIDE INDICATORS
        self.performance_system['total simulation time'] =

        return performance_agents, performance_system

    def visualize_performance(self, performance_agents, performance_system):

