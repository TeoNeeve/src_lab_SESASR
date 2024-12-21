from lab06.Path_Planners.dijkstra import Dijkstra


class AStar(Dijkstra):

    def node_insertion(self, new_node, current, goal):
        """Method to insert a new node in the frontier"""
        new_node.g = self.cost_so_far[current] + self.cost_g(current, new_node)
        if new_node not in self.cost_so_far or new_node.g < self.cost_so_far[new_node]:
            self.cost_so_far[new_node] = new_node.g
            new_node.h = self.cost_h(new_node, goal)
            self.frontier.put(new_node, new_node.f)
            self.came_from[new_node] = current

    def cost_h(self, current, goal):
        return self.movements_class.heuristic_cost(current.position, goal.position)

    def reset(self):
        return super().reset()
