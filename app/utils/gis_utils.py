import numpy as np


def get_polygon_square_area(coordinates: list) -> float:
    coordinates = np.array(coordinates)
    lon = [co[0] for co in coordinates]
    lat = [co[1] for co in coordinates]
    return (np.max(lon) - np.min(lon)) * (np.max(lat) - np.min(lat))


def change_lon_lat(coordinates: list) -> list:
    coordinates = np.array(coordinates)
    s = coordinates.shape
    assert len(s) >= 2 and s[-1] == 2
    if len(s) == 2:
        return coordinates[:, ::-1].tolist()
    elif len(s) == 3:
        return coordinates[:, :, ::-1].tolist()
    elif len(s) == 4:
        return coordinates[:, :, :, ::-1].tolist()


def change_lon_lat_any(coordinates: list) -> list:
    coordinates = np.array(coordinates)
    s = coordinates.shape
    if len(s) >= 2 and s[-1] == 2:
        return change_lon_lat(coordinates.tolist())
    else:
        co_out = []
        for i, i_co in enumerate(coordinates):
            i_co = np.array(i_co)
            s = i_co.shape
            if len(s) >= 2 and s[-1] == 2:
                co_out.append(change_lon_lat(i_co.tolist()))
            else:
                co_out.append([])
                for j, j_co in enumerate(i_co):
                    j_co = np.array(j_co)
                    s = j_co.shape
                    if len(s) >= 2 and s[-1] == 2:
                        co_out[i].append(change_lon_lat(j_co.tolist()))
                    else:
                        raise ValueError("Unparsable shape of input coordinates.")
        return co_out


def get_outer_ring_polygons(coordinates: list) -> list:
    coordinates = np.array(coordinates)

    out = []
    for i_coordinates in coordinates:
        i_coordinates = np.array(i_coordinates)
        while not (len(i_coordinates.shape) == 2 and i_coordinates.shape[-1] == 2):
            i_coordinates = np.array(i_coordinates[0])
        out.append(i_coordinates.tolist())
    return out


def get_largest_outer_ring_polygon(coordinates: list) -> list:
    coordinates = np.array(coordinates)

    largest_polygon_size = 0
    largest_polygon = None
    for i_coordinates in coordinates:
        i_coordinates = np.array(i_coordinates)
        while not (len(i_coordinates.shape) == 2 and i_coordinates.shape[-1] == 2):
            i_coordinates = np.array(i_coordinates[0])
        size = get_polygon_square_area(i_coordinates)
        if largest_polygon_size < size:
            largest_polygon = i_coordinates.tolist()
            largest_polygon_size = size
    return largest_polygon
