"""
This file contains the DistributedAgent class that can be used to implement individual planning.

Code in this file is just provided as guidance, you are free to deviate from it.
"""
import time as timer
from single_agent_planner import a_star, get_sum_of_cost

class DistributedAgent(object):
    """DistributedAgent object to be used in the distributed planner."""

    def __init__(self, my_map, start, goal, heuristics, agent_id):
        """
        my_map   - list of lists specifying obstacle positions
        starts      - (x1, y1) start location
        goals       - (x1, y1) goal location
        heuristics  - heuristic to goal location
        """
        self.my_map = my_map
        self.start = start
        self.goal = goal
        self.id = agent_id
        self.heuristics = heuristics
        self.path = self.find_solution()
        self.random = 0

    def find_solution(self, constraints=[]):
        """ Finds paths for all agents from their start locations to their goal locations."""

        start_time = timer.time()

        result = a_star(self.my_map, self.start, self.goal, self.heuristics, self.id, constraints)

        if result is None:
            raise BaseException('No solutions')


        self.CPU_time = timer.time() - start_time



        return result