import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial import distance
import csv

# Function to fit a line
def fit_line(points):
    x, y = points[:, 0], points[:, 1]
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]  # Slope of the line

# Function to fit a circle
def fit_circle(points):
    def circle_func(coords, a, b, r):
        x, y = coords
        return np.sqrt((x - a) ** 2 + (y - b) ** 2) - r

    x, y = points[:, 0], points[:, 1]
    initial_guess = [np.mean(x), np.mean(y), np.std(x)]
    coeffs, _ = curve_fit(circle_func, (x, y), np.zeros(len(points)), p0=initial_guess)
    return coeffs

# Function to identify symmetry
def identify_symmetry(points):
    centroid = np.mean(points, axis=0)
    mirrored_points = 2 * centroid - points
    dist = distance.directed_hausdorff(points, mirrored_points)[0]
    return dist < 1e-3  # Threshold for symmetry

# Function to complete curves
def complete_curve(points):
    filled_points = []
    for i in range(len(points) - 1):
        filled_points.append(points[i])
        if np.linalg.norm(points[i + 1] - points[i]) > 1:
            filled_points.extend(np.linspace(points[i], points[i + 1], num=10))
    filled_points.append(points[-1])
    return np.array(filled_points)

# Function to read CSV file
def read_csv(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

# Function to plot shapes
def plot(paths_XYs):
    fig, ax = plt.subplots(tight_layout=True, figsize=(8, 8))
    colours = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    plt.show()

# Function to check if a shape is a rectangle
def is_rectangle(points):
    x, y = points[:, 0], points[:, 1]
    x_diff = np.diff(x)
    y_diff = np.diff(y)
    min_length = min(len(x_diff[::2]), len(x_diff[1::2]), len(y_diff[::2]), len(y_diff[1::2]))
    return np.allclose(x_diff[::2][:min_length], -x_diff[1::2][:min_length]) and \
           np.allclose(y_diff[::2][:min_length], -y_diff[1::2][:min_length])

# Function to check if a shape is a rounded rectangle
def is_rounded_rectangle(points):
    return len(points) == 8 and is_rectangle(points)

# Function to check if a shape is a regular polygon
def is_regular_polygon(points):
    vectors = np.diff(points, axis=0)
    vectors = np.vstack([vectors, points[0] - points[-1]])  # Close the polygon
    sides = np.linalg.norm(vectors, axis=1)
    norms_product = np.clip(sides * np.roll(sides, -1), a_min=1e-10, a_max=None)
    dot_products = np.einsum('ij,ij->i', vectors, np.roll(vectors, -1, axis=0))
    angles = np.degrees(np.arccos(np.clip(dot_products / norms_product, -1.0, 1.0)))
    angles = np.nan_to_num(angles)
    return np.allclose(sides, sides[0]) and np.allclose(angles, angles[0])

# Function to write CSV file
def write_csv(csv_path, processed_paths):
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i, shape in enumerate(processed_paths):
            for j, path in enumerate(shape):
                for k, point in enumerate(path):
                    writer.writerow([i, j, k, point[0], point[1]])

# Example usage
csv_path = 'isolated.csv'
path_XYs = read_csv(csv_path)

processed_paths = []
for shape in path_XYs:
    processed_shape = []
    for path in shape:
        line_coeffs = fit_line(path)
        circle_coeffs = fit_circle(path)
        symmetry = identify_symmetry(path)
        completed_points = complete_curve(path)
        rectangle = is_rectangle(completed_points)
        rounded_rectangle = is_rounded_rectangle(completed_points)
        regular_polygon = is_regular_polygon(completed_points)
        processed_shape.append(completed_points)
    processed_paths.append(processed_shape)

# Write processed shapes to CSV
output_csv_path = csv_path.replace('.csv', '_sol.csv')
write_csv(output_csv_path, processed_paths)

# Visualize original shapes
plot(path_XYs)

# Visualize processed shapes
plot(processed_paths)
