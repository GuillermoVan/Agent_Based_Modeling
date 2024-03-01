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
#Function 2: input = map of 0s and 1s, current location all agents -> if agent detected, then output = detected agent's object

#DETECT CONFLICT + COMMUNICATION
#Function 3: input = agent object -> output = True/False conflict detection
#if conflict detected, shortest path without passing conflict point, slowest path 'wins' -> output = path update for agent who 'lost'

#LATER ON WE WILL UITBREIDEN TO MULTIPLE AGENTS IN SAME SCOPE AND SEE HOW THAT WORKS


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
        # T.B.D.

    def define_scope(self, result, timestep, agentID1, scope_rad=2):
        complete_map = np.zeros_like(self.my_map)
        for row in self.my_map:
            for cell in self.my_map[row]:
                if not cell:
                    complete_map[row][cell] = 1

        neighbors = [(i, j) for i in range(-scope_rad, scope_rad + 1) for j in range(-scope_rad, scope_rad + 1)]

        (center_row, center_col) = result[agentID1][timestep]
        scope_map = np.copy(complete_map)
        for i in range(len(complete_map)):
            for j in range(len(complete_map[i])):
                distance = abs(center_row - i) + abs(center_col - j)  # Manhattan distance
                if distance > scope_rad:
                    scope_map[i][j] = 0
                # if any(abs(row - i) > scope_rad or abs(col - j) > scope_rad for row, col in neighbors):
                #     scope_map[i][j] = 0

        return scope_map

        # for dr, dc in neighbors:
        #     for row in complete_map:
        #         for column in complete_map[row]:
        #             new_row, new_col = row + dr, column + dc
        #             if 0 <= new_row < len(complete_map) and 0 <= new_col<=len(complete_map[row]):
        #                 if complete_map[row][column] == :
        #
        #                     complete_map[new_row][new_col] =

        # for all_agents in range(self.num_of_agents):
        #     if agentID1 != all_agents:
        #         if result[agentID1][timestep] == result[all_agents][timestep]:
        #             complete_map
        #


    def find_solution(self):
        """
        Finds paths for all agents from start to goal locations. 
        
        Returns:
            result (list): with a path [(s,t), .....] for each agent.
        """
        # Initialize constants       
        start_time = timer.time()
        result = []
        self.CPU_time = timer.time() - start_time
        
        
        # Create agent objects with DistributedAgent class
        for i in range(self.num_of_agents):
            newAgent = DistributedAgent(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i)

        
        
        # Print final output
        print("\n Found a solution! \n")
        print("CPU time (s):    {:.2f}".format(self.CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(result)))  # Hint: think about how cost is defined in your implementation
        print(result)
        
        return result  # Hint: this should be the final result of the distributed planning (visualization is done after planning)