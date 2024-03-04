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
        result = []
        self.CPU_time = timer.time() - start_time

        # Create agent objects with DistributedAgent class and initiate their independent paths
        for i in range(self.num_of_agents):
            newAgent = DistributedAgent(self.my_map, self.starts[i], self.goals[i], self.heuristics[i], i)
            result_i = newAgent.find_solution()
            result.append(result_i)
            self.distributed_agents.append(newAgent)

        time = 0
        arrived = []

        #while not all agents have reached their target
        while len(arrived) < len(self.distributed_agents):
            for agent_1 in distributed_agents:
                scope_map = define_scope(self, result, time, agent_1.id, scope_rad=2)
                agents_in_scope = detect_agent_in_scope(self, agent_1, scope_map, time)
                for agent_2 in agents_in_scope:
                    result = conflict(self, agent_1, agent_2, time)

                if result[agent_1.id][time] == self.goals[agent_1.id]:
                    arrived.append(agent_1)

            time += 1

        return result  # Hint: this should be the final result of the distributed planning (visualization is done after planning)

    def define_scope(self, result, timestep, agentID1, scope_rad=2):
        complete_map = [[0 if cell else 1 for cell in row] for row in self.my_map]

        center_row, center_col = result[agentID1][timestep]

        # Create a copy of the complete_map
        scope_map = np.copy(complete_map)

        # Iterate over the cells and set values based on the scope_rad
        for i in range(len(complete_map)):
            for j in range(len(complete_map[i])):
                distance = abs(center_row - i) + abs(center_col - j)  # Manhattan distance
                if distance > scope_rad:
                    scope_map[i][j] = 0

        return scope_map

    def conflict(self, agent_1, agent_2, time):
        for i in range(3):  # Communicate 3 time steps
            avoidance = False
            timestep = time + (i + 1)

            while avoidance == False:  # Only continue when collision is avoided
                if agent_1.path[time] == agent_2.path[time]:  # Collision at timestep
                    """ Construct alternative (temporary) paths for the colliding agents"""
                    constraint_temp = []

                    constraint_temp.append({'positive': False,
                                            'negative': True,
                                            'agent': agent_1.id,
                                            'loc': agent_1.path[timestep],
                                            'timestep': timestep
                                            })

                    constraint_temp.append({'positive': False,
                                            'negative': True,
                                            'agent': agent_2.id,
                                            'loc': agent_2.path[timestep],
                                            'timestep': timestep
                                            })

                    path_1 = agentID_1.find_solution(constraints=constraint_temp)

                    path_2 = agentID_2.find_solution(constraints=constraint_temp)

                    """ Agent with the longest detour gets to keep its original path."""

                    if path_1 >= path_2:
                        agent_2.path = [agent_2.path[:timestep], path_2[:]]

                    """ The changed path cannot be changed back to include the collision point at the same timestep."""

                        constraints.append({'positive': False,
                                            'negative': True,
                                            'agent': agent_2.id,
                                            'loc': agent_1[timestep],
                                            'timestep': timestep
                                            })
                    else:
                        agent_1.path = [agent_1.path[:timestep], path_1[:]]

                        constraints.append({'positive': False,
                                            'negative': True,
                                            'agent': agent_1.id,
                                            'loc': agent_2.path[timestep],
                                            'timestep': timestep
                                            })
                else:
                    avoidance = True

        return result, constraints

# DETECT AGENT
# Function 2: input = map of 0s and 1s, current location all agents -> if agent detected, then output = detected agent's object

    def detect_agent_in_scope(self, checking_agent, map, time):
        detected_agents = []

        for agent in self.distributed_agents:
            curr_loc = agent.path[time]
            if map[curr_loc[1]][curr_loc[0]] == 1 and agent.id != checking_agent.id:
                detected_agents.append(agent)

        return detected_agents
