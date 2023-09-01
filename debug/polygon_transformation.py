import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
def prune_points(points, th=0.1):
    points_pruned = [points[0]]
    for i in range(1, len(points)):
        x1, y1 = points_pruned[-1]
        x2, y2 = points[i]
        dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dist > th:
            points_pruned.append(points[i])
    return points_pruned

def interpolate_points(ps, pe):
    xs, ys = ps
    xe, ye = pe
    points = []
    dx = xe - xs
    dy = ye - ys
    if dx != 0:
        scale = dy / dx
        if xe > xs:
            x_interpolated = list(range(ceil(xs), floor(xe) + 1))
        else:
            x_interpolated = list(range(floor(xs), ceil(xe) - 1, -1))
        for x in x_interpolated:
            y = ys + (x - xs) * scale
            points.append([x, y])
    if dy != 0:
        scale = dx / dy
        if ye > ys:
            y_interpolated = list(range(ceil(ys), floor(ye) + 1))
        else:
            y_interpolated = list(range(floor(ys), ceil(ye) - 1, -1))
        for y in y_interpolated:
            x = xs + (y - ys) * scale
            points.append([x, y])
    if xe > xs:
        points = sorted(points, key=lambda x: x[0])
    else:
        points = sorted(points, key=lambda x: -x[0])
    return points

def apply_rotation(polygon, angle):
    # Apply rotation transformation to the polygon
    centroid = np.mean(polygon, axis=0)
    rotated_polygon = (polygon - centroid) @ np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return rotated_polygon + centroid

def interpolate_polygon(polygon):
    # points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    points = polygon
    points_interpolated = []
    points_interpolated.append(points[0])
    for i in range(0, len(points) - 1):
        points_i = interpolate_points(points[i], points[i + 1])
        points_interpolated += points_i
        points_interpolated.append(points[i + 1])
    points_interpolated = prune_points(points_interpolated)
    polygon_interpolated = np.array(points_interpolated)
    return polygon_interpolated.reshape(-1, 2)

import numpy as np
import matplotlib.pyplot as plt

def arrange_polygon(polygon):
    # Arrange the polygon to have the starting point closest to the origin
    centroid = np.mean(polygon, axis=0)
    sorted_indices = np.argsort(np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0]))
    arranged_polygon = polygon[sorted_indices]
    return arranged_polygon

def apply_translation(polygon, tx, ty):
    # Apply translation transformation to the polygon
    translated_polygon = polygon + np.array([tx, ty])
    return translated_polygon

def apply_shearing(polygon, sx, sy):
    # Apply shearing transformation to the polygon
    sheared_polygon = polygon + np.array([polygon[:, 1] * sx, polygon[:, 0] * sy]).T
    return sheared_polygon

def apply_noise(polygon, magnitude):
    # Add random noise to the polygon
    noisy_polygon = polygon + np.random.uniform(-magnitude, magnitude, size=polygon.shape)
    return noisy_polygon

def apply_flip(polygon, axis):
    # Apply flip transformation to the polygon
    flipped_polygon = np.copy(polygon)
    flipped_polygon[:, axis] = 1 - flipped_polygon[:, axis]
    return flipped_polygon

def normalize_polygon(polygon, scale):
    # Normalize the polygon coordinates to the range of [0, 1] using the provided scale
    normalized_polygon = polygon / scale
    return np.clip(normalized_polygon, 0, 1)

def plot_polygon(polygon, connect_points=True):
    # Plot the polygon with optional connection between start and end points
    if connect_points:
        polygon = np.vstack((polygon, polygon[0]))
    plt.plot(polygon[:, 0], polygon[:, 1], '-o')


# Example usage
polygon = np.array([[20, 25], [30, 35], [40, 45], [50, 55], [50, 54], [30, 21], [20, 15], [10, 5], [1, 20], [20, 23]])
scale = np.array([500, 500])  # Image scale [x, y]

# Arrange the polygon to have the starting point closest to the origin
arranged_polygon = arrange_polygon(polygon)

rotation_angle = np.pi / 4  # in radians
rotated_polygon = apply_rotation(arranged_polygon, rotation_angle)
# Apply translation
translated_polygon = apply_translation(arranged_polygon, 100, 100)

# Apply shearing
sheared_polygon = apply_shearing(arranged_polygon, -0.5, -0.5)

# Apply noise
noisy_polygon = apply_noise(arranged_polygon, 20)

# Apply flip
flipped_polygon = apply_flip(arranged_polygon, axis=0)

# Normalize the transformed polygons
interpolated_polygon = normalize_polygon(interpolate_polygon(arranged_polygon), scale)
arranged_polygon = normalize_polygon(arranged_polygon, scale)
translated_polygon = normalize_polygon(translated_polygon, scale)
sheared_polygon = normalize_polygon(sheared_polygon, scale)
rotated_polygon = normalize_polygon(rotated_polygon, scale)
noisy_polygon = normalize_polygon(noisy_polygon, scale)
flipped_polygon = normalize_polygon(flipped_polygon, scale)
# Plot the original and normalized transformed polygons
plt.figure(figsize=(16, 14))
# plt.subplot(2, 3, 1)
# plot_polygon(arranged_polygon)
# plt.title('Original Polygon')
plt.subplot(2, 3, 1)
plot_polygon(interpolated_polygon)
plt.title('Dense Polygon')

plt.subplot(2, 3, 2)
plot_polygon(translated_polygon)
plt.title('Translated Polygon')

plt.subplot(2, 3, 3)
plot_polygon(sheared_polygon)
plt.title('Sheared Polygon')

plt.subplot(2, 3, 4)
plot_polygon(noisy_polygon)
plt.title('Noisy Polygon')

plt.subplot(2, 3, 5)
plot_polygon(rotated_polygon)
plt.title('rotated Polygon')

plt.subplot(2, 3, 6)
plot_polygon(arranged_polygon)
plt.title('Normalized Polygon')

plt.tight_layout()
plt.show()