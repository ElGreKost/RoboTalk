import math

import numpy as np
from numpy.random import multivariate_normal


# OBSERVATION FUNCTIONS
def polar_distance_to_goal(current_pos, goal_pos):
    """
    Calculates the change in radius (d_r) and change in the angle
    (d_theta) between two positions in polar coordinates.
    Arguments:
        current_pos & goal_pos are in Euclidean space (x,z)
    Returns:
        A tuple containing d_r (change in radius) and d_theta (change in angle) as floats.
    """
    curr_x, curr_z = current_pos
    goal_x, goal_z = goal_pos

    # Calculate current radius and angle
    curr_radius = math.sqrt(curr_x ** 2 + curr_z ** 2)  # Consider z for 3D space
    curr_angle = math.atan2(curr_x, curr_z)  # pitch

    # Calculate goal radius and angle
    goal_radius = math.sqrt(goal_x ** 2 + goal_z ** 2)  # Consider z for 3D space
    goal_angle = math.atan2(goal_x, goal_z)  # Adjust for quadrant handling

    # Calculate change in radius (d_r)
    d_r = abs(goal_radius - curr_radius)

    # Calculate change in angle (d_theta)
    # Handle potential angle wrapping around the circle (e.g., (-pi) to pi)
    d_theta = goal_angle - curr_angle
    d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi  # Normalize to -pi to pi range

    return d_r, d_theta


# REWARD FUNCTIONS
def gaussian_eucl_distance_reward(desired_pos, current_pos, multiplier=10, scale=1.0, width_factor=0.5,
                                  height_factor=2.0):
    """Calculates Gaussian distance reward with adjustable width and height factors."""
    goal_distance = np.linalg.norm([current_pos[0] / width_factor, current_pos[1] / height_factor] - desired_pos)
    gauss_dist = lambda x: np.exp(-(x ** 2) / scale)

    if goal_distance < 0.05:
        reward = 10 * multiplier
    elif goal_distance > 0.9:
        reward = - multiplier * goal_distance
    else:
        reward = multiplier * gauss_dist(goal_distance)

    # reward = 3 * multiplier if goal_distance < 0.1 else multiplier * gauss_dist(goal_distance)
    return reward


def linear_eucl_distance_reward(desired_pos, current_pos, max_rew=100, scale=20):
    goal_distance = np.linalg.norm(current_pos - desired_pos)
    if goal_distance < 0.05:
        reward = max_rew
    else:
        reward = 10 - scale * goal_distance

    return reward


def multivariate_normal_weighted(mean=(0, 0), cov=None, weight_x=1.0, weight_y=1.0):
    return multivariate_normal(mean=mean, cov=cov if cov is not None else ([weight_x ** 2, 0], [0, weight_y ** 2])).pdf

# add this to class if you need to find the drone's velocity projection norm towards the goal
# def calc_vel_to_goal(self) -> float:
#     curr_pos_vec = np.array(self.get_drone_pos())
#     goal_pos_vec = np.array(self.goal_pos)
#     drone_vel_vec = np.array(self.get_drone_vel())
#
#     # Vector from current position to goal
#     direction_vector = goal_pos_vec - curr_pos_vec
#
#     # Projection of drone   's velocity on the direction towards the goal
#     proj_on_goal = drone_vel_vec @ direction_vector / np.linalg.norm(direction_vector)
#
#     proj_on_goal_norm = np.linalg.norm(proj_on_goal)
#
#     return proj_on_goal_norm
