import random
import math
import matplotlib.pyplot as plt

def get_distance_ratio(point0, point1) -> float:
    return math.sqrt((point0[0]-point1[0])**2 + (point0[1]-point1[1])**2)
def get_random_point_in_circle(radius : float):
    """
    :param radius: radius of a circle
    :return: position (x,y) of random point in circle
    """
    angle = random.uniform(0, 2 * math.pi)

    distance = random.uniform(0, radius)

    return distance, angle

def get_point_from_polar_notation(pos_x: float, pos_y: float, distance:float, angle:float):
    """
    :param pos_x: x of center of circle
    :param pos_y: y of center of circle
    :param distance: radius of circle
    :param angle: angle in polar notation
    :return: position (x,y) of point in circle
    """

    x = pos_x + distance * math.cos(angle)
    y = pos_y + distance * math.sin(angle)

    return x, y


class Particle:
    def __init__(self, circles):
        """
        :param circles: list of tuples (center_x, center_y, radius) representing bounds of each point
        """
        # Points stored as (distance_from_center, angle)
        self.points = [get_random_point_in_circle(circle[2]) for circle in circles]
        # velocities stored as (distance_vel, angle_vel)
        self.velocity = [(random.uniform(-circle[2], circle[2]), random.uniform(-2 * math.pi, 2 * math.pi))for circle in circles]
        self.best_local_solution = list(self.points)
        self.lowest_local_distance = float('inf')

    def update_velocity(self, best_global_solution, omega=0.7, phi_p=0.2, phi_g=0.2):
        r_l = random.random()
        r_g = random.random()

        vel_local = [(phi_p * r_l * (p_best[0] - p[0]), phi_p * r_l * (p_best[1] - p[1])) for p_best, p in
                     zip(self.best_local_solution, self.points)]
        vel_global = [(phi_g * r_g * (g_best[0] - p[0]), phi_g * r_g * (g_best[1] - p[1])) for g_best, p in
                      zip(best_global_solution, self.points)]
        self.velocity = [(omega * v[0] + v_l[0] + v_g[0], omega * v[1] + v_l[1] + v_g[1]) for v, v_l, v_g in
                         zip(self.velocity, vel_local, vel_global)]

    def update_points(self, circles, learning_rate=0.1):
        for i in range(len(circles)):
            temp_distance = self.points[i][0] + learning_rate * self.velocity[i][0]
            temp_angle = self.points[i][1] + learning_rate * self.velocity[i][1]
            if temp_distance > circles[i][2]:
                temp_distance = circles[i][2]
            elif temp_distance < 0:
                temp_distance = 0
            if temp_angle > 2 * math.pi:
                temp_angle %= 2 * math.pi
            elif temp_angle < 0:
                temp_angle %= 2 * math.pi
            self.points[i] = (temp_distance, temp_angle)

    def update_best(self, circles, start_point, best_global_solution, lowest_global_distance):
        """
        :param circles: list of circles in standard notation ( center_x, center_y, radius)
        :param start_point: starting point (x, y)
        :param best_global_solution: list of best points
        :param lowest_global_distance: the distance between points in best global solution
        :return:
        """
        result_length = 0
        result_length += get_distance_ratio(start_point, get_point_from_polar_notation(circles[0][0], circles[0][1],
                                                                                       self.points[0][0],
                                                                                       self.points[0][1]))
        for i in range(len(self.points) - 1):
            result_length += get_distance_ratio(
                get_point_from_polar_notation(circles[i][0], circles[i][1], self.points[i][0], self.points[i][1]),
                get_point_from_polar_notation(circles[i + 1][0], circles[i + 1][1], self.points[i + 1][0],
                                              self.points[i + 1][1]))
        result_length += get_distance_ratio(start_point, get_point_from_polar_notation(circles[-1][0], circles[-1][1],
                                                                                       self.points[-1][0],
                                                                                       self.points[-1][1]))

        if result_length < self.lowest_local_distance:
            self.best_local_solution = list(self.points)
            self.lowest_local_distance = result_length

            if result_length < lowest_global_distance:
                best_global_solution = list(self.points)
                lowest_global_distance = result_length

        return best_global_solution, lowest_global_distance
def test():
    n = 5 # number of circles
    k = 20 # swarm count

    circles = [(random.uniform(5,35), random.uniform(5,35), random.uniform(1,4)) for _ in range(n)]
    start_point = (random.uniform(5,35), random.uniform(5,35))

    particles = []
    global_best, gb_len = [], float(math.inf)

    for j in range(k):
        particles.append(Particle(circles))
        global_best, gb_len = particles[-1].update_best(circles,start_point,global_best,gb_len)

    initial_points = [get_point_from_polar_notation(circles[i][0], circles[i][1], particles[0].points[i][0], particles[0].points[i][1]) for i in range(len(circles))]

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plot initial configuration
    axs[0].set_title("Initial configuration")
    for circle in circles:
        circle1 = plt.Circle((circle[0], circle[1]), circle[2], color='b', fill=False)
        axs[0].add_patch(circle1)
    axs[0].scatter(start_point[0], start_point[1], color='r')  # plot the starting point
    axs[0].plot(*zip(*([start_point] + initial_points + [start_point])), color='g')  # plot the initial points with lines connecting them
    axs[0].scatter(*zip(*initial_points), color='g')  # plot the initial points
    axs[0].axis('scaled')

    print(gb_len)
    for _ in range(10000):
        for j in range(k):
            particles[j].update_velocity(global_best)
            particles[j].update_points(circles)
            global_best,gb_len = particles[j].update_best(circles,start_point,global_best,gb_len)
    print(gb_len)

    final_points = [get_point_from_polar_notation(circles[i][0], circles[i][1], particles[0].points[i][0], particles[0].points[i][1]) for i in range(len(circles))]

    # Plot final configuration
    axs[1].set_title("Final configuration")
    for circle in circles:
        circle1 = plt.Circle((circle[0], circle[1]), circle[2], color='b', fill=False)
        axs[1].add_patch(circle1)
    axs[1].scatter(start_point[0], start_point[1], color='r')  # plot the starting point
    axs[1].plot(*zip(*([start_point] + final_points + [start_point])), color='g')  # plot the final points with lines connecting them
    axs[1].scatter(*zip(*final_points), color='g')  # plot the final points
    axs[1].axis('scaled')

    plt.show()

test()
