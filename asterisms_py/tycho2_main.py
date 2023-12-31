# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/01_polars+pytorch.ipynb.

# %% auto 0
__all__ = ['resultdf', 'ScoreItem', 'Points', 'convert_to_cartesian', 'calculate_distances', 'distance_from_magnitude',
           'distance_from_magnitude_tensor', 'tetrahedron_score', 'square_score', 'measure_squareness_old',
           'measure_squareness', 'mass_score_triangle_torch', 'transform_radecmag_from_numpy',
           'global_normalize_tensor', 'radec_normalize_tensor', 'mag_score', 'score_triangle',
           'stars_for_point_and_radius', 'stars_for_center_and_radius', 'get_grid_points', 'get_grid_point_by_idx',
           'get_region', 'get_center', 'ra_to_hms']

# %% ../nbs/01_polars+pytorch.ipynb 2
import polars as pl
import numpy as np
import torch
import math
import time
import heapq
from typing import List, Any
import tqdm
import pickle
import pyarrow as pa
from dataclasses import dataclass, field

# %% ../nbs/01_polars+pytorch.ipynb 5
@dataclass(order=True)
class ScoreItem:
    score: float
    region: int=field(compare=False)
    item: Any=field(compare=False)
    
@dataclass(order=True)
class Points:
    points: Any=field(compare=False)
    

def convert_to_cartesian(distances, ra, dec):
    x = distances * torch.cos(dec) * torch.cos(ra)
    y = distances * torch.cos(dec) * torch.sin(ra)
    z = distances * torch.sin(dec)
    return torch.stack((x, y, z), dim=1)

def calculate_distances(coords):
    dist_matrix = torch.cdist(coords, coords)
    unique_distances = torch.unique(dist_matrix)
    return unique_distances[unique_distances > 0]

def distance_from_magnitude(m, M):
    return 10**((m - M + 5) / 5)

def distance_from_magnitude_tensor(m: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    return 10**((m - M + 5) / 5)

# one of the possibilities
def tetrahedron_score(coords):
    distances = calculate_distances(coords)
    print(distances)
    mean_distance = torch.mean(distances)
    std_dev = torch.std(distances)

    # Assuming std_dev is small enough, the score will be close to 1
    # Otherwise, it will be closer to 0
    score = math.exp(-std_dev.item() / mean_distance.item())
    return score

def square_score(tensor):
    """Return a measure of how close the points in a tensor are to forming a square,
    as well as the standard deviation of their brightness."""
    
    # Calculate pairwise distances based on the x and y coordinates (first two dimensions)
    spatial_distances = torch.pdist(tensor, p=2)
    print(spatial_distances)
    
    # Sort the distances
    sorted_distances = torch.sort(spatial_distances)[0]
    
    # Take the 4 smallest distances and compute their standard deviation
    std_of_smallest_4 = sorted_distances[:4].std().item()
    
    # # Calculate the standard deviation of the brightness (third dimension)
    # brightness_std = tensor[:, 2].std().item()
    
    return std_of_smallest_4

def measure_squareness_old(tensor):
    """
    :param points: A tensor of shape (4, 2) representing the 4 2D points
    :return: A float indicating the squareness. Closer to 1 means more square.
    """
    
    # Calculate pairwise distances based on the x and y coordinates (first two dimensions)
    spatial_distances = torch.pdist(tensor, p=2)

    # Sort the distances
    distances = torch.sort(spatial_distances)[0]
    
    # Mean of sides and diagonals
    mean_sides = torch.mean(distances[:4])
    mean_diagonal = torch.mean(distances[4:])
    
    # Squareness measure
    squareness = mean_diagonal / mean_sides
    
    # Normalize with sqrt(2) to get values closer to 1 for squares
    return abs(1 - (squareness / torch.sqrt(torch.tensor(2.0)).item()))

def measure_squareness(tensor):
    # Calculate pairwise distances based on the x and y coordinates (first two dimensions)
    spatial_distances = torch.pdist(tensor, p=2)

    # Sort the distances
    distances = torch.sort(spatial_distances)[0]
   
    # Compute the standard deviation for the four shortest distances and the two longest distances
    std_sides = torch.std(distances[:4])
    std_diagonal = torch.std(distances[4:])
    
    # Ideally, for a perfect square, the standard deviations would be 0
    # We use exp(-x) as a measure to get values close to 1 for low standard deviations
    side_uniformity = torch.exp(-std_sides)
    diagonal_uniformity = torch.exp(-std_diagonal)
    
    # Mean of sides and diagonals
    mean_sides = torch.mean(distances[:4])
    mean_diagonal = torch.mean(distances[4:])
    
    # Squareness measure based on side to diagonal ratio
    squareness_ratio = mean_sides / mean_diagonal
    
    # Combine all the measures
    # Normalize with sqrt(2) to get values closer to 1 for squares
    final_squareness = (squareness_ratio / torch.sqrt(torch.tensor(2.0)).item()) * side_uniformity * diagonal_uniformity
    
    return abs(1 - final_squareness.item())

def mass_score_triangle_torch(points_tensor, device='cpu'):
    points_tensor = points_tensor.to(device)  # Transfer tensor to GPU if available
    idx_combinations = torch.combinations(torch.arange(points_tensor.shape[0]), r=3)
                                          
    p1, p2, p3 = points_tensor[idx_combinations[:, 0]], points_tensor[idx_combinations[:, 1]], points_tensor[idx_combinations[:, 2]]

    a = torch.linalg.norm(p2 - p1, dim=1)
    b = torch.linalg.norm(p3 - p2, dim=1)
    c = torch.linalg.norm(p1 - p3, dim=1)
    
    mean = (a + b + c) / 3
    std_dev = torch.sqrt(((a - mean)**2 + (b - mean)**2 + (c - mean)**2) / 3)
    
    scores = torch.where(mean != 0, std_dev / mean, torch.tensor([1.], device=mean.device))
    
     # Stack the points instead of flattening them
    points_combined = torch.stack([p1, p2, p3], dim=1)
    
    return scores, points_combined

# go from tycho2 to xyz coords
def transform_radecmag_from_numpy(stars):
    torch_tensors = [torch.from_numpy(star) for star in stars]
    zeroes = torch.zeros(len(torch_tensors[2]))
    print("one ", torch_tensors)
    torch_tensors[2] = distance_from_magnitude_tensor(torch_tensors[2], zeroes)
    print("two", torch_tensors)
    coords = convert_to_cartesian(*torch_tensors)
    return coords

def global_normalize_tensor(tensor):
    """Normalize a tensor based on its global min and max values. Also works for multiple tensors"""
    global_min = torch.min(tensor)
    global_max = torch.max(tensor)
    
    normalized = (tensor - global_min) / (global_max - global_min)
    return normalized
    
def radec_normalize_tensor(tensors):
    """Normalize tensors based on their global min and max values, excluding the 3rd column."""

    # Concatenate tensors while excluding the 3rd column
    # Drop the 3rd column from each tensor
    tensor = tensors[:, :2]

    # Compute global min and max excluding the 3rd column
    global_min = torch.min(tensor)
    global_max = torch.max(tensor)

    # Normalize tensors using the computed global min and max
    normalized = (tensor - global_min) / (global_max - global_min)
    return normalized

def mag_score(tensor):
    # Computing the standard deviation
    # stdev = t[:, 2].std()
    max = tensor[:, 2].max()
    min = tensor[:, 2].min()
    return max - min

def score_triangle(tensor):    
    # Calculate pairwise distances based on the x and y coordinates (first two dimensions)
    spatial_distances = torch.pdist(tensor, p=2)
    
    # Normalize with sqrt(2) to get values closer to 1 for squares
    return torch.std(spatial_distances)

def stars_for_point_and_radius(df, point, radius, max_mag):
    """ point is in the corner, not the center """
    ra, dec = point
    minra = ra
    maxra = ra + radius
    mindec = dec
    maxdec = dec + radius
    return df.filter((pl.col("RAmdeg") < maxra) & (pl.col("RAmdeg") > minra) & (pl.col("DEmdeg") < maxdec) & (pl.col("DEmdeg") > mindec) & (pl.col("Vmag") <= max_mag))

def stars_for_center_and_radius(df, center, radius, max_mag):
    ra, dec = center
    minra = ra - radius/2
    maxra = ra + radius/2
    mindec = dec - radius/2
    maxdec = dec + radius/2
    return df.filter((pl.col("RAmdeg") < maxra) & (pl.col("RAmdeg") > minra) & (pl.col("DEmdeg") < maxdec) & (pl.col("DEmdeg") > mindec) & (pl.col("Vmag") <= max_mag))


def get_grid_points(min_dec=-90, max_dec=90):
    RA_values = [ra for ra in range(0, 361, 4)]  # Increment by 4 for a 2-degree radius
    Dec_values = [dec for dec in range(min_dec, max_dec+1, 4)]  # Increment by 4 for a 2-degree radius
    grid_points = [(ra, dec) for ra in RA_values for dec in Dec_values]
    return grid_points

def get_grid_point_by_idx(idx):
    gp = get_grid_points()
    return gp[idx]

def get_region(df, idx, radius, max_mag, min_dec=-90, max_dec=90):
    center = get_grid_points(min_dec, max_dec)[idx]
    return stars_for_center_and_radius(df, center, radius, max_mag)

def get_center(dftycho, center, radius, max_mag):
    return stars_for_point_and_radius(dftycho, center, radius, max_mag)

def ra_to_hms(ra):
    if ra < 0.0:
        ra = ra + 360
    mm, hh = math.modf(ra / 15.0)
    _, mm = math.modf(mm * 60.0)
    ss = round(_ * 60.0)
    return hh, mm, ss

resultdf = pl.DataFrame({
    "score": pl.Float64,
    "region": pl.Int64,
    "item": []
})


    

