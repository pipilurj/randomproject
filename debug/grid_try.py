import numpy as np


def convert_to_grid(points, grid_size):
    grid_numbers = np.zeros(points.shape[0], dtype=int)
    for i, point in enumerate(points):
        x, y = point
        grid_x = min(int(x * grid_size), grid_size-1)
        grid_y = min(int(y * grid_size), grid_size-1)
        grid_number = grid_y + grid_x * grid_size
        grid_numbers[i] = grid_number
    return grid_numbers

def convert_to_continuous(grid_numbers, grid_size):
    points = np.zeros((grid_numbers.shape[0], 2))
    for i, grid_number in enumerate(grid_numbers):
        grid_y = grid_number % grid_size
        grid_x = grid_number // grid_size
        x = (grid_x + 0.5) / grid_size
        y = (grid_y + 0.5) / grid_size
        points[i] = [x, y]
    return points
import random
# Example usage
# points = np.array([[0.2, 0.3], [0.8, 0.6], [0.4, 0.1], [1, 0.]])
# grid_size = 1000
#
# # Convert points to grid numbers
# grid_numbers = convert_to_grid(points, grid_size)
# print("Grid numbers:", grid_numbers)
#
# # Convert grid numbers back to points
# converted_points = convert_to_continuous(grid_numbers, grid_size)
# print("Converted points:", converted_points)
# print(random.randint(5, 5))
print(convert_to_continuous(np.array([4, 7, 8]), 3))