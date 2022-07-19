from typing import Union, List, Set

import numpy as np


def elevate_pts(area_pts, new_z):
    new_area_pts = []
    for p in area_pts:
        new_p = list(p)
        new_p[1] = new_z
        new_area_pts.append(tuple(new_p))
    return new_area_pts


def offset_pts(pts: Union[List, Set], *, offset_x=0, offset_h=0, offset_y=0):
    new_pts = [(x + offset_x, h + offset_h, y + offset_y) for x, h, y in pts]
    if isinstance(pts, set):
        return set(new_pts)
    return new_pts


def box_area(x0, z0, y, x1, z1):
    area_points = []
    for i in np.linspace(x0, x1, num=int(abs(x1 - x0) + 1)):
        for j in np.linspace(z0, z1, num=int(abs(z1 - z0) + 1)):
            area_points.append((i, y, j))
    return area_points


def line_area(x0, z0, y, x1, z1):
    if not (x0 == x1 or z0 == z1):
        raise ValueError('The line is not axis-aligned:' + str((x0, x1, y, z0, z1)))
    area_points = []

    for i in np.linspace(x0, x1, num=int(abs(x1 - x0) + 1)):
        for j in np.linspace(z0, z1, num=int(abs(z1 - z0) + 1)):
            area_points.append((i, y, j))
    return area_points


def get_box_size(box):
    return abs(box[0] - box[3]), abs(box[1] - box[4])


def offset_boxes(bl, h, offset_x, offset_y):
    bl_new = []
    if isinstance(h, list):
        assert len(bl) == len(h)
    for i, b in enumerate(bl):
        if len(b) == 4:
            x0, y0, x1, y1 = b
        else:
            assert (len(b) == 5)
            x0, y0, _, x1, y1 = b
        x0 += offset_x
        x1 += offset_x
        y0 += offset_y
        y1 += offset_y
        if isinstance(h, list):
            bl_new.append((x0, y0, h[i], x1, y1))
        else:
            bl_new.append((x0, y0, h, x1, y1))
    return bl_new


def get_area_points_with_p5(area_points):
    positions_init = set()
    for p in area_points:

        # handling stud with 0.5 offset
        for offset in [-0.5, 0.5]:
            positions_init.add(tuple(np.array(p) + [offset, 0, 0]))
            positions_init.add(tuple(np.array(p) + [0, 0, offset]))
            for offset2 in [-0.5, 0.5]:
                positions_init.add(tuple(np.array(p) + [offset, 0, offset2]))

        positions_init.add(tuple(p))
    return positions_init


def get_plane_occ_points(area_points, height):
    plane_occ_init = set()
    for p in area_points:
        for k in range(height):
            for offset_x in [-0.25, 0.25]:
                for offset_y in [-0.25, 0.25]:
                    for offset_z in [-0.25, 0.25]:
                        p_offset = np.array(p) + [offset_x, offset_y + k, offset_z]
                        plane_occ_init.add(tuple(p_offset))
    return plane_occ_init


from sklearn.neighbors import NearestNeighbors


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist
