from typing import List

import numpy as np


def get_polygon_rectangle_area(coordinates: List[float]) -> float:
    """Fits a rectangle (exactly N/S and E/W aligned) around a polygon
    and computes the area of the rectangle.

    Args:
        coordinates (List[float]): Polygon coordinates

    Returns:
        float: Area of rectangle around polygon
    """
    coordinates = np.array(coordinates)
    lon = [co[0] for co in coordinates]
    lat = [co[1] for co in coordinates]
    return (np.max(lon) - np.min(lon)) * (np.max(lat) - np.min(lat))


def get_largest_outer_ring_polygon(coordinates: list) -> List[float]:
    """From a list of polygons, return the one with the largest outer rectangle area

    Args:
        coordinates (list): List of polygons

    Returns:
        List[float]: Largest with the largest outer rectangle
    """
    coordinates = np.array(coordinates)

    largest_polygon_size = 0
    largest_polygon = None
    for i_coordinates in coordinates:
        i_coordinates = np.array(i_coordinates)
        while not (len(i_coordinates.shape) == 2 and i_coordinates.shape[-1] == 2):
            i_coordinates = np.array(i_coordinates[0])
        size = get_polygon_rectangle_area(i_coordinates)
        if largest_polygon_size < size:
            largest_polygon = i_coordinates.tolist()
            largest_polygon_size = size
    return largest_polygon
