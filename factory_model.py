"""Factory of the future.

* Raw Materials at stock.
* Machine converts raw materials to product.
* Robots transport raw material from stock to machine and finished products.
    from machine to packagin.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from pypaths import astar
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import SingleGrid


class Store(Agent):
    """A store that find nearest robots to transport goods."""

    def __init__(self, unique_id, factory, pos):
        """Initialize store."""
        super().__init__(unique_id, factory)
        self.pos = pos
        self.factory = factory
        self.raw_materials = 100
        self.orders = 0

    def step(self):
        """Conduct actions of a store for one step.

        * If an order is in queue, then call for the nearest robot to take raw_materials.
            * Currently, one order is equal to one raw_material.
        * If a robot is at the store, give it raw_material and ask it go to machine.
        """
        self.summon_robot()
        self.handle_robot_at_location()
        # Debug
        print("[STORE] raw_materials: {}  orders: {}".format(self.raw_materials, self.orders))

    def summon_robot(self):
        """If raw materials are not empty, then call the nearest robot."""
        if self.orders > 0:
            if self.raw_materials > 0:
                nearest_robot = self.factory.find_nearest_aimless_robot(self.pos)
                if nearest_robot:
                    nearest_robot.destination = self.pos
                    self.orders -= 1
                else:
                    print("WARNING: No aimless robots at the moment, will summon again next step.")
            else:
                print("WARNING: Not enough raw materials to process order.")

    def handle_robot_at_location(self):
        """If a robot is at the store, then transfer raw materials onto it and send it to machine."""
        robot_at_location = self.factory.find_robot_at_position(self.pos)
        if robot_at_location:
            if robot_at_location.destination == self.pos:
                robot_at_location.destination = self.factory.machine.pos
                robot_at_location.raw_materials += 1
                self.raw_materials -= 1
            else:
                print("ERROR: Store, this should not happen.")


class Machine(Agent):
    """A machine that find nearest robots to transport goods."""

    def __init__(self, unique_id, factory, pos):
        """."""
        super().__init__(unique_id, factory)
        self.pos = pos
        self.factory = factory
        self.raw_materials = 0
        self.products = 0

    def step(self):
        """Conduct actions of a machine for one step.

        * If no robots at the machine and a product is ready then summon a robot.
        * If a robot is at the machine:
            * and if robot has raw materials then transfer them to the machine.
            * and if the machine has a product then transfer it to the robot and ask it go to packaging.
        * If the machine has raw materials, then manufacture a product.
        """
        self.summon_robot()
        self.handle_robot_at_location()
        self.manufacture_product()
        # Debug.
        print("[MACHINE] raw_materials: {}  products:{}".format(self.raw_materials, self.products))

    def summon_robot(self):
        """If products are available then find nearest to take the products to packaging."""
        if self.products > 0:
            nearest_robot = self.factory.find_nearest_aimless_robot(self.pos)
            if nearest_robot:
                nearest_robot.destination = self.pos
            else:
                print("WARNING: No aimless robots at the moment, will summon again next step.")

    def handle_robot_at_location(self):
        """If a robot shows up at the machine with raw materials or if the machine has products.

        * If robot has only raw materials, then machine takes raw materials and releases the robot.
        * If robot has only products, then machine transfers products to robot and sends it to packaging dept.
        * If robot has raw materials and machine has products, then machine takes raw materials, then transfers to it
            and then sends it to packaging dept.
        * If robot does not have raw materials and machine has no products, then it is not an expected situation.
        """
        robot_at_location = self.factory.find_robot_at_position(self.pos)
        if robot_at_location:
            if robot_at_location.destination == self.pos:
                if robot_at_location.raw_materials > 0 and self.products == 0:
                    self.raw_materials += 1
                    robot_at_location.raw_materials -= 1
                    robot_at_location.destination = None

                elif robot_at_location.raw_materials == 0 and self.products > 0:
                    self.products -= 1
                    robot_at_location.products += 1
                    robot_at_location.destination = self.factory.packaging.pos

                elif robot_at_location.raw_materials > 0 and self.products > 0:
                    self.raw_materials += 1
                    robot_at_location.raw_materials -= 1
                    self.products -= 1
                    robot_at_location.products += 1
                    robot_at_location.destination = self.factory.packaging.pos

                else:
                    print("ERROR: Machine-1, this should not happen.")
            else:
                print("ERROR: Machine-2, this should not happen.")

    def manufacture_product(self):
        """If there are raw_materials, then manufacture a product."""
        if self.raw_materials > 0:
            self.raw_materials -= 1
            self.products += 1


class Packaging(Agent):
    """A packaging agent that receives finished products from robots."""

    def __init__(self, unique_id, factory, pos):
        """."""
        super().__init__(unique_id, factory)
        self.pos = pos
        self.factory = factory
        self.products = 0

    def step(self):
        """Conduct actions of packaging for one step.

        If a robot is at the machine, it transfers the product into packaging.
        """
        self.handle_robot_at_location()

    def handle_robot_at_location(self):
        """Transfer the product from the robot into packaging."""
        robot_at_location = self.factory.find_robot_at_position(self.pos)
        if robot_at_location:
            if robot_at_location.destination == self.pos:
                if robot_at_location.products > 0:
                    self.products += robot_at_location.products
                    robot_at_location.products = 0
                    robot_at_location.destination = None
                else:
                    print("ERROR: Packaging-1, this should not occur.")
            else:
                print("ERROR: Packaging-2, this should not occur.")


class Robot(Agent):
    """Factory robot that does all the transportation of raw materials and products."""

    def __init__(self, unique_id, factory):
        """Initialize a robot."""
        super().__init__(unique_id, factory)
        self.factory = factory
        self.destination = None
        self.raw_materials = 0
        self.products = 0

    def step(self):
        """Conduct actions of packaging for one step.

        * If destination is None, random walk.
        * If destination, move to destination.
        """
        if self.destination is None:
            if self.products == 0 and self.raw_materials == 0:
                self.random_walk()
            else:
                print("ERROR: Robot, this should not happen.")
        else:
            self.move_to_destination()

    def move_to_destination(self):
        """Move towards destination."""
        next_pos = self.factory.find_next_position_towards_destination(self.pos, self.destination)
        self.factory.grid.move_agent(self, next_pos)

    def random_walk(self):
        """Do a random walk from current position, but keep away from the departments."""
        next_pos = self.factory.find_next_position_for_random_walk(self.pos)
        print("[ROBOT] Before Loiter curr_pos: {}  next_pos: {}".format(self.pos, next_pos))
        self.factory.grid.move_agent(self, next_pos)
        print("[ROBOT] After Loiter curr_pos: {}".format(self.pos))


class Factory(Model):
    """The Factory model that maintains the state of the whole factory."""

    def __init__(self, grid_w, grid_h, n_robots):
        """Initialize factory."""
        # Initialize.
        self.orders = 0
        self.n_robots = n_robots
        self.scheduler = RandomActivation(self)
        self.grid = SingleGrid(grid_w, grid_h, torus=False)
        self.init_astar()
        # Initialize departments.
        self.machine = Machine("machine", self, self.grid.find_empty())
        self.store = Store("store", self, self.grid.find_empty())
        self.packaging = Packaging("packaging", self, self.grid.find_empty())
        self.dept_positions = [self.machine.pos, self.store.pos, self.packaging.pos]
        # Initialize robots.
        for i in range(self.n_robots):
            # Create robot.
            r = Robot(i, self)
            # Initialize random location.
            pos = self.grid.find_empty()
            self.grid.place_agent(r, pos)
            # Register with scheduler.
            self.scheduler.add(r)
        # Initialize visualization.
        plt.ion()

    def add_order(self):
        """Increment the number of orders to the factory."""
        self.orders += 1

    def step(self):
        """Advance the factory by one step."""
        # Step through factory. Check for orders.
        if self.orders > 0:
            self.store.orders += 1
            self.orders -= 1
        # Step through departments.
        self.store.step()
        self.machine.step()
        self.packaging.step()
        # Step through robots.
        self.scheduler.step()
        # Visualize.
        self.visualize()

    def init_astar(self):
        """Initialize a-star resources so that it doesn't have to calculated for each robot.

        Initialized in such a way that:
            * A diagonal paths are allowed.
            * The path calculated takes into account all obstacles in the grid.
        """
        def get_empty_neighborhood(pos):
            """A sub function to calculate empty neighbors of a point for a-star."""
            neighbors = self.grid.get_neighborhood(pos=pos, moore=True)
            return [n for n in neighbors if self.grid.is_cell_empty(n)]
        # Initialize a path finder object once for the entire factory.
        self.path_finder = astar.pathfinder(neighbors=get_empty_neighborhood,
                                            distance=astar.absolute_distance,
                                            cost=astar.fixed_cost(1))

    def find_nearest_aimless_robot(self, pos):
        """Find the nearest aimless robot to a given position in the factory."""
        def is_aimless(robot, pos):
            """Check if the robot satisfied aimless condition."""
            if robot.destination is None:
                return True
            else:
                return False

        aimless_robots = [robot for robot in self.scheduler.agents if is_aimless(robot, pos)]
        if len(aimless_robots) != 0:
            robot_distances = [astar.absolute_distance(pos, robot.pos) for robot in aimless_robots]
            nearest_index = np.argmin(robot_distances)
            return aimless_robots[nearest_index]
        else:
            return None

    def find_robot_at_position(self, pos):
        """Find robot that is at a given location in the factory that is not busy."""
        for robot in self.scheduler.agents:
            if robot.pos == pos:
                return robot
        return None

    def find_next_position_towards_destination(self, curr_pos, dest_pos):
        """Find the next empty position to move in the direction of the destination."""
        n_steps, path = self.path_finder(curr_pos, dest_pos)  # Handles non-empty locations.
        # NOTE: We cannot find a valid path to the destination when:
        #   1) The destination has an another robot located inside it, which also occurs when curr_pos and
        #       dest_pos are the same.
        #   2) The path is entirely blocked.
        #   In these cases we return the next position to be the curr_pos, in order to wait until things
        #   clear up.
        if n_steps is None or n_steps <= 0:  # No valid path to destination
            next_pos = curr_pos
            print("[MOVE] Warning: No path to destination from {} --> {}".format(curr_pos, dest_pos))
        # This mean there's a valid path to destination.
        else:
            # index 0, is the curr_pos, index 1 is the next position.
            next_pos = path[1]
        return next_pos

    def find_next_position_for_random_walk(self, curr_pos):
        """Find a valid location for a robot to just randomly walk into."""
        def is_pos_empty(pos):
            """A sub function if a cell is empty for random walking."""
            if self.grid.is_cell_empty(pos) and pos not in self.dept_positions:
                return True
            else:
                return False
        neighborhood = self.grid.get_neighborhood(curr_pos, moore=True)
        empty_neighborhood = [n for n in neighborhood if is_pos_empty(n)]
        if len(empty_neighborhood) > 0:
            next_index = np.random.randint(len(empty_neighborhood))
            next_pos = empty_neighborhood[next_index]
        else:
            next_pos = curr_pos
        return next_pos

    def visualize(self):
        """A chess board type visualization."""
        def heatmap(a):
            cMap = ListedColormap(['grey', 'black', 'green', 'orange', 'red', 'blue'])
            sns.heatmap(a, vmin=0, vmax=6, cmap=cMap, linewidths=1)
            plt.pause(0.3)
            plt.clf()

        g = np.zeros((self.grid.height, self.grid.width), dtype=int)
        g[self.store.pos] = 3
        g[self.machine.pos] = 4
        g[self.packaging.pos] = 5
        for robot in self.scheduler.agents:
            if robot.destination is None:
                g[robot.pos] = 1
            else:
                g[robot.pos] = 2

        heatmap(g)


if __name__ == '__main__':
    factory = Factory(20, 20, 5)

    for i in range(2):
        factory.add_order()

    for i in range(100):
        factory.step()
