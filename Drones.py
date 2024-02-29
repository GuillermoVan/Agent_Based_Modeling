# import time as timer
# from single_agent_planner import compute_heuristics, a_star, get_sum_of_cost
#
# class DronesSolver(object):
#     """A planner that plans for each drone independently."""
#
#     def __init__(self):
#         """my_map   - list of lists specifying obstacle positions
#                 starts      - [(x1, y1), (x2, y2), ...] list of start locations
#                 goals       - [(x1, y1), (x2, y2), ...] list of goal locations
#                 """
#
#         self.my_map = my_map
#         self.starts = starts
#         self.goals = goals
#         self.num_of_agents = len(goals)
#
#         self.CPU_time = 0
#
#         # compute heuristics for the low-level search
#         self.heuristics = []
#         for goal in self.goals:
#             self.heuristics.append(compute_heuristics(my_map, goal))
#
#         def find_solution(self):
#             """ Finds paths for all agents from their start locations to their goal locations."""
#
#             start_time = timer.time()
#             result = []
#             constraints = []
#
#             """ Priority for longer detour."""
#
#
#             for i in range(self.num_of_agents):
#                 """ Finds initial path for all agents from their start locations to their goal locations."""
#
#                 path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
#                               i, constraints)
#                 if path is None:
#                     raise BaseException('No solutions')
#
#                 result.append(path)
#
#             self.CPU_time = timer.time() - start_time
#
#             """" Start simulation and check for potential future collisions."""
#             agents_array = [agent for _, agent in self.agents.items()]
#             for i in range(0, len(agents_array)):
#                 for j in range(i + 1, len(agents_array)):
#                     d1 = agents_array[i]
#                     d2 = agents_array[j]
#                     pos1 = np.array(d1.center)
#                     pos2 = np.array(d2.center)
#                     if np.linalg.norm(pos1 - pos2) < 2 * np.sqrt(2):


#########################################################
def conflict(self, result, agentID_1, agentID_2, curr_time):
    for i in range(3): # Communicate 3 time steps
        avoidance = False
        timestep = curr_time + (i + 1)

        while avoidance = False: # Only continue when collision is avoided
            if result[agentID_1][timestep] == result[agentID_2][timestep]: # Collision at timestep
                """ Construct alternative (temporary) paths for the colliding agents"""
                constraint_temp = []

                constraint_temp.append({'positive': False,
                                        'negative': True,
                                        'agent': agentID_1,
                                        'loc': result[agentID_1][timestep],
                                        'timestep': timestep
                                        })

                constraint_temp.append({'positive': False,
                                        'negative': True,
                                        'agent': agentID_2,
                                        'loc': result[agentID_1][timestep],
                                        'timestep': timestep
                                        })

                path_1 = a_star(self.my_map, self.starts[agentID_1], self.goals[agentID_1],
                                self.heuristics[agentID_1],
                                agentID_1, constraint_temp)

                path_2 = a_star(self.my_map, self.starts[agentID_2], self.goals[agentID_2], self.heuristics[agentID_2],
                                agentID_2, constraint_temp)

                """ Agent with the longest detour gets to keep its original path."""

                if path_1 >= path_2:
                    result[agentID_2] = path_2

                """ The changed path cannot be changed back to include the collision point at the same timestep."""

                    constraints.append({'positive': False,
                                        'negative': True,
                                        'agent': agentID_2,
                                        'loc': result[agentID_1][timestep],
                                        'timestep': timestep
                                        })
                else:
                    result[agentID_1] = path_1

                    constraints.append({'positive': False,
                                        'negative': True,
                                        'agent': agentID_1,
                                        'loc': result[agentID_1][timestep],
                                        'timestep': timestep
                                        })
            else:
                avoidance = True

    return result, constraints


                # Moet naar path finding functie
                # scope = []
                # for j in [-2, 0, 2]:
                #     for k in [-2, 0, 2]:
                #         if my_map[curr['loc'] + j][curr['loc'] + k] == False: # Surrounding grid tile is not obstacle
                #             scope.append((curr['loc'] + j, curr['loc'] + k))





