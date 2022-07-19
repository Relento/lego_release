import collections
import functools
import json
import math
import os
from typing import List, Set, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trimesh.transformations as tr

from bricks.utils import elevate_pts, offset_pts
from common import ROOT

brick_rotation_candidates = [tr.quaternion_about_axis(math.pi / 2 * i, [0, 1, 0]) for i in range(4)]

brick_annotation = None
EXCLUDE_LSHAPE = os.getenv('EXCLUDE_LSHAPE', '')


def get_brick_annotation(bid):
    global brick_annotation
    if brick_annotation is None:
        import plistlib
        with open(os.path.join(ROOT, 'data/parts_info.plist'), 'rb') as f:
            brick_annotation = plistlib.load(f)['Part List']

    # handling name aliases
    if bid == '4073':
        bid = '6141'
    elif bid == '3070':
        bid = '3070a'

    bid = bid + '.dat'
    if bid == '2357.dat':
        return 'L shape brick'
    elif bid in brick_annotation:
        return brick_annotation[bid]['Part Name']
    else:
        return 'Unknown brick type'


consider_symmetry = True


def get_consider_symmetry():
    global consider_symmetry
    return consider_symmetry


def set_consider_symmetry(v):
    global consider_symmetry
    consider_symmetry = v


# get mapping from ldr color code to RGBA:
color_map = {}


def get_color_map():
    global color_map
    if not color_map:
        with open('data/ldraw_color_config.json') as f:
            color_map = json.load(f)
    return color_map


# generate coordinates of lego boxes putting on the ground
def lego_box_coords(l, h, w):
    center = np.array([(l - 1) / 2, 0, (w - 1) / 2])
    pts = []
    for i in range(l):
        for j in range(h):
            for k in range(w):
                pts.append(np.array([i, j, k]) - center)
    return pts


def simple_brick_info(l, h, w):
    brick_pts = lego_box_coords(l, h, w)
    stud_pts = list((np.array(p) + [0, 1, 0]) for p in brick_pts if p[1] == h - 1)
    astud_pts = list(p for p in brick_pts if p[1] == 0)
    if l >= 2 and w >= 2:
        center = np.array([(l - 2) / 2, 0, (w - 2) / 2])
        for i in range(l - 1):
            for j in range(w - 1):
                astud_pts.append(np.array([i, 0, j]) - center)
    if w == 1:
        center = np.array([(l - 2) / 2, 0, 0])
        for i in range(l - 1):
            astud_pts.append(np.array([i, 0, 0]) - center)
    elif h == 1:
        center = np.array([0, 0, (w - 2) / 2])
        for i in range(w - 1):
            astud_pts.append(np.array([0, 0, j]) - center)

    base_contour = []
    for sign in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
        base_contour.append(np.array([l / 2 * sign[0], 0, w / 2 * sign[1]]))

    return brick_pts, stud_pts, astud_pts, base_contour


def get_brick_bbox_xyz(occ_pts: List[np.ndarray]):
    # WARNINGS: this is (l, h, w) not (l, w, h)!!!!

    # WARNINGS: if brick occ points are removed after initialization
    # the bounding cube size might change
    b_occ_np = np.stack(occ_pts)
    l = b_occ_np[:, 0].max() - b_occ_np[:, 0].min() + 1
    h = b_occ_np[:, 1].max() - b_occ_np[:, 1].min() + 1
    w = b_occ_np[:, 2].max() - b_occ_np[:, 2].min() + 1
    # return l, w, h
    return l, h, w


class BasicBrickInfo:
    def __init__(self,
                 bid: str,
                 # List of 3d points.
                 occ_pts: List[np.array],
                 stud_pts: List[np.array],
                 astud_pts: List[np.array],
                 # For detecting base collision.
                 base_contour: List[np.array],
                 height: int,
                 rotations: List[int],
                 ):
        self.bid = bid
        self.occ_pts = occ_pts
        self.stud_pts = stud_pts
        self.astud_pts = astud_pts
        self.base_contour = base_contour
        self.height = height
        self.rotations = rotations

        # WARNINGS: do not modify fields above after initialization
        #  otherwise base_bbox_size will be incorrect

        # compute base size based on contour
        corners_np = np.stack(base_contour)
        self.base_bbox_size = (corners_np[:, 0].max() - corners_np[:, 0].min(),
                               corners_np[:, 2].max() - corners_np[:, 2].min())
        self.bbox_xyz = get_brick_bbox_xyz(self.occ_pts)

    def get_info(self):
        return self.occ_pts, self.stud_pts, self.astud_pts, self.base_contour

    def get_height(self):
        return self.height

    def get_valid_rotations(self):
        if get_consider_symmetry():
            return self.rotations
        else:
            return [0, 1, 2, 3]

    def get_annotation(self):
        pass

    def __repr__(self):
        return f'bid: {self.bid}\n' + \
               f'occ_pts: {self.occ_pts}\n' + \
               f'stud_pts: {self.stud_pts}\n' + \
               f'astud_pts: {self.astud_pts}\n' + \
               f'base_contour: {self.base_contour}\n' + \
               f'height: {self.height}\n' + \
               f'rotations: {self.rotations}'


class SimpleBrickInfo(BasicBrickInfo):
    """
       Define simple rectangular bricks by its size
    """

    def __init__(self, bid, l, w, h, is_base=False):
        occ_pts, stud_pts, astud_pts, base_contour = simple_brick_info(l, h, w)
        if l == w:
            rotations = [0]
        else:
            rotations = [0, 1]
        super().__init__(bid, occ_pts, stud_pts, astud_pts, base_contour,
                         height=h, rotations=rotations)
        self.size = (l, w, h)
        self.is_base = is_base

    def get_size(self):
        # FIXME: size is (l, w, h) but bbox size is (l, h, w)
        return self.size


class TileBrickInfo(SimpleBrickInfo):
    def __init__(self, bid, l, w):
        super().__init__(bid, l, w, 1)
        self.stud_pts = []


brick_list = {}


class SlopeBrickInfo(BasicBrickInfo):
    def __init__(self, bid, l, w, h, stud_pts, rotations, occ_pts=None):
        cube_occ_pts, _, cube_astud_pts, cube_base_contour = simple_brick_info(l, h, w)
        super().__init__(
            bid=bid,
            occ_pts=occ_pts if occ_pts is not None else cube_occ_pts,
            stud_pts=stud_pts,
            astud_pts=cube_astud_pts,
            base_contour=cube_base_contour,
            height=h,
            rotations=rotations,
        )


def get_slope_bricks():
    def remove_np_arrays(l: List[np.array], arrs: List[np.array]):
        for a in arrs:
            ind = 0
            size = len(l)
            while ind != size and not np.array_equal(l[ind], a):
                ind += 1
            if ind != size:
                l.pop(ind)
            else:
                raise ValueError(f'array {str(a)} not found in list.')

    bricks = []

    b_15571 = SlopeBrickInfo(
        bid='15571',
        l=2,
        w=1,
        h=3,
        stud_pts=[],
        rotations=[0, 1, 2, 3]
    )
    bricks.append(b_15571)

    brick_4460b = SlopeBrickInfo(
        bid='4460b',
        l=1,
        w=2,
        h=9,
        stud_pts=[np.array([0, 9, -0.5])],
        rotations=[0, 1, 2, 3]
    )
    remove_np_arrays(brick_4460b.occ_pts, [
        np.array([0, i, 0.5]) for i in range(4, 9)
    ])
    bricks.append(brick_4460b)

    b_4286 = SlopeBrickInfo(
        bid='4286',
        l=1,
        w=3,
        h=3,
        stud_pts=[np.array([0, 3, -1])],
        rotations=[0, 1, 2, 3]
    )
    remove_np_arrays(b_4286.occ_pts, [np.array([0, 2, 0]), np.array([0, 1, 1]), np.array([0, 2, 1])])
    bricks.append(b_4286)

    b_3298 = SlopeBrickInfo(
        bid='3298',
        l=2,
        w=3,
        h=3,
        stud_pts=[np.array([-0.5, 3, -1]), np.array([0.5, 3, -1])],
        rotations=[0, 1, 2, 3]
    )
    remove_np_arrays(b_3298.occ_pts, [
        np.array([-0.5, 2, 0]), np.array([-0.5, 1, 1]), np.array([-0.5, 2, 1]),
        np.array([0.5, 2, 0]), np.array([0.5, 1, 1]), np.array([0.5, 2, 1]),
    ])
    bricks.append(b_3298)

    b_3099 = SlopeBrickInfo(
        bid='3039',
        l=2,
        w=2,
        h=3,
        stud_pts=[np.array([-0.5, 3, -0.5]), [0.5, 3, -0.5]],
        rotations=[0, 1, 2, 3],
    )
    bricks.append(b_3099)

    remove_np_arrays(b_3099.occ_pts, [np.array([-0.5, 2, 0.5]), np.array([0.5, 2, 0.5])])
    b_3660 = SlopeBrickInfo(
        bid='3660',
        l=2,
        w=2,
        h=3,
        stud_pts=[np.array([-0.5, 3, -0.5]), np.array([-0.5, 3, 0.5]), np.array([0.5, 3, -0.5]),
                  np.array([0.5, 3, 0.5])],
        rotations=[0, 1, 2, 3]
    )
    remove_np_arrays(b_3660.occ_pts, [np.array([0.5, 0, 0.5]), np.array([-0.5, 0, 0.5])])
    remove_np_arrays(b_3660.astud_pts, [np.array([0.5, 0, 0.5]), np.array([-0.5, 0, 0.5]), np.array([0, 0, 0])])
    b_3660.base_contour = [np.array([-1, 0, 0]), np.array([1, 0, 0]), np.array([1, 0., -1]), np.array([-1, 0, -1])]
    bricks.append(b_3660)

    b_3665 = SlopeBrickInfo(
        bid='3665',
        l=1,
        w=2,
        h=3,
        stud_pts=[np.array([0, 3, -0.5]), np.array([0, 3, 0.5])],
        rotations=[0, 1, 2, 3]
    )
    remove_np_arrays(b_3665.occ_pts, [np.array([0, 0, 0.5])])
    remove_np_arrays(b_3665.astud_pts, [np.array([0, 0, 0.5])])
    b_3665.base_contour = [np.array([-0.5, 0, 0]), np.array([0.5, 0, 0]), np.array([0.5, 0, -1]),
                           np.array([-0.5, 0, -1])]
    bricks.append(b_3665)

    b_37352 = SlopeBrickInfo(
        bid='37352',
        l=2,
        w=1,
        h=3,
        stud_pts=[],
        rotations=[0, 1, 2, 3]
    )
    bricks.append(b_37352)

    b_60481 = SlopeBrickInfo(
        bid='60481',
        l=1,
        w=2,
        h=6,
        stud_pts=[np.array([0, 6, -0.5])],
        rotations=[0, 1, 2, 3]
    )
    remove_np_arrays(b_60481.occ_pts, [
        np.array([0, i, 0.5]) for i in range(3, 6)
    ])
    bricks.append(b_60481)

    b_85984 = SlopeBrickInfo(
        bid='85984',
        l=2,
        w=1,
        h=2,
        stud_pts=[],
        rotations=[0, 1, 2, 3]
    )
    # remove_np_arrays(b_85984.occ_pts, [np.array([0, 2, 0.5])])
    bricks.append(b_85984)

    b_3040b = SlopeBrickInfo(
        bid='3040b',
        l=1,
        w=2,
        h=3,
        stud_pts=[np.array([0, 3, -0.5])],
        rotations=[0, 1, 2, 3],
    )
    remove_np_arrays(b_3040b.occ_pts, [np.array([0, 2, 0.5])])
    bricks.append(b_3040b)

    for brick in bricks:
        old_bbox_xyz = brick.bbox_xyz
        bbox_xyz = get_brick_bbox_xyz(brick.occ_pts)
        assert old_bbox_xyz == bbox_xyz, (brick, old_bbox_xyz, bbox_xyz)

    return {
        b.bid: b for b in bricks
    }


def get_misc_bricks():
    bricks = []
    occ_pts, stud_pts, astud_pts, base_contour = simple_brick_info(1, 3, 1)
    b_87087 = BasicBrickInfo(
        bid='87087',
        occ_pts=occ_pts,
        stud_pts=stud_pts,
        astud_pts=astud_pts,
        base_contour=base_contour,
        height=3,
        rotations=[0, 1, 2, 3],
    )
    bricks.append(b_87087)

    b_6231 = SimpleBrickInfo('6231', 1, 1, 3)
    b_6231.stud_pts = []
    b_6231.rotations = [0, 1, 2, 3]
    bricks.append(b_6231)

    b_54200 = SimpleBrickInfo('54200', 1, 1, 2)
    b_54200.stud_pts = []
    b_54200.rotations = [0, 1, 2, 3]
    bricks.append(b_54200)
    return {
        b.bid: b for b in bricks
    }


def init_brick_list():
    simple_brick_list = {
        # Simple Bricks
        '3024': SimpleBrickInfo('3024', 1, 1, 1),
        '3005': SimpleBrickInfo('3005', 1, 1, 3),
        '3023': SimpleBrickInfo('3023', 2, 1, 1),
        '3004': SimpleBrickInfo('3004', 2, 1, 3),
        '3623': SimpleBrickInfo('3623', 3, 1, 1),
        '3622': SimpleBrickInfo('3622', 3, 1, 3),
        '3010': SimpleBrickInfo('3010', 4, 1, 3),
        '3666': SimpleBrickInfo('3666', 6, 1, 1),
        '3009': SimpleBrickInfo('3009', 6, 1, 3),
        '3460': SimpleBrickInfo('3460', 8, 1, 1),
        '3008': SimpleBrickInfo('3008', 8, 1, 3),
        '3022': SimpleBrickInfo('3022', 2, 2, 1),
        '3003': SimpleBrickInfo('3003', 2, 2, 3),
        '3021': SimpleBrickInfo('3021', 3, 2, 1),
        '3020': SimpleBrickInfo('3020', 4, 2, 1),
        '3001': SimpleBrickInfo('3001', 4, 2, 3),
        '3795': SimpleBrickInfo('3795', 6, 2, 1),
        '2456': SimpleBrickInfo('2456', 6, 2, 3),
        '3034': SimpleBrickInfo('3034', 8, 2, 1),
        '3007': SimpleBrickInfo('3007', 8, 2, 3),
        '3036': SimpleBrickInfo('3036', 8, 6, 1, is_base=True),
        '41539': SimpleBrickInfo('41539', 8, 8, 1, is_base=True),
        '3958': SimpleBrickInfo('3958', 6, 6, 1, is_base=True),
        '3033': SimpleBrickInfo('3033', 10, 6, 1, is_base=True),
        '3002': SimpleBrickInfo('3002', 3, 2, 3),
        '2445': SimpleBrickInfo('2445', 12, 2, 1),
        '3030': SimpleBrickInfo('3030', 10, 4, 1, is_base=True),
        '3031': SimpleBrickInfo('3031', 4, 4, 1),
        '3032': SimpleBrickInfo('3032', 6, 4, 1),
        '3035': SimpleBrickInfo('3035', 8, 4, 1),
        '3710': SimpleBrickInfo('3710', 4, 1, 1),
        '3832': SimpleBrickInfo('3832', 10, 2, 1),
        '4073': SimpleBrickInfo('4073', 1, 1, 1),
        '4477': SimpleBrickInfo('4477', 10, 1, 1),
        '6112': SimpleBrickInfo('6112', 12, 1, 3),
        '3062b': SimpleBrickInfo('3062b', 1, 1, 3),
        '3245a': SimpleBrickInfo('3245a', 2, 1, 6),
        '4032a': SimpleBrickInfo('4032a', 2, 2, 1),
        '60479': SimpleBrickInfo('60479', 12, 1, 1),
        '4282': SimpleBrickInfo('4282', 16, 2, 1),
        # '772': SimpleBrickInfo('772', 2, 1, 6), same with 3245a
        '22886': SimpleBrickInfo('22886', 2, 1, 9),
        '3941': SimpleBrickInfo('3941', 2, 2, 3),
        '2877': SimpleBrickInfo('2877', 2, 1, 3),
        # Note that this brick's side face is not vertical and needs special treatment
        '59900': SimpleBrickInfo('59900', 1, 1, 3),
        '3029': SimpleBrickInfo('3029', 12, 4, 1, is_base=True),
        # '6141': SimpleBrickInfo('6141', 1, 1, 1), # same with 4073
    }

    tile_brick_list = {
        '3070': TileBrickInfo('3070', 1, 1),
        '2431': TileBrickInfo('2431', 4, 1),
        '4150': TileBrickInfo('4150', 2, 2),
        '6636': TileBrickInfo('6636', 6, 1),
        '3068b': TileBrickInfo('3068b', 2, 2),
        '3069b': TileBrickInfo('3069b', 2, 1),
        '3070b': TileBrickInfo('3070b', 1, 1),
        '63864': TileBrickInfo('63864', 3, 1),
        '4162': TileBrickInfo('4162', 8, 1),
    }

    # 0.5 center brick
    p5_offset_brick_list = {}
    bid = '3794a'
    stud_pts = [np.array([0, 1, 0])]
    occ_pts, _, astud_pts, base_contour = simple_brick_info(2, 1, 1)
    p5_offset_brick_list[bid] = BasicBrickInfo(
        bid=bid,
        occ_pts=occ_pts,
        stud_pts=stud_pts,
        astud_pts=astud_pts,
        base_contour=base_contour,
        height=1,
        rotations=[0, 1],
    )

    bid = '87580'
    stud_pts = [np.array([0, 1, 0])]
    occ_pts, _, astud_pts, base_contour = simple_brick_info(2, 1, 2)
    p5_offset_brick_list[bid] = BasicBrickInfo(
        bid=bid,
        occ_pts=occ_pts,
        stud_pts=stud_pts,
        astud_pts=astud_pts,
        base_contour=base_contour,
        height=1,
        rotations=[0],
    )

    base_contour_L = list(map(np.array,
                              [[-0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5], [0.5, 0, 1.5],
                               [-0.5, 0, 1.5]]
                              ))

    occ_pts_2357 = []
    for i in range(3):
        occ_pts_2357 += list(map(np.array, [[0, i, 0], [1, i, 0], [0, i, 1]]))

    lshape_brick_list = {
        '2420': BasicBrickInfo(
            bid='2420',
            occ_pts=list(map(np.array, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])),
            stud_pts=list(map(np.array, [[0, 1, 0], [1, 1, 0], [0, 1, 1]])),
            astud_pts=list(map(np.array, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])),
            base_contour=base_contour_L,
            height=1,
            rotations=[0, 1, 2, 3]
        ),
        '2357': BasicBrickInfo(
            bid='2357',
            occ_pts=occ_pts_2357,
            stud_pts=list(map(np.array, [[0, 3, 0], [1, 3, 0], [0, 3, 1]])),
            astud_pts=list(map(np.array, [[0, 0, 0], [1, 0, 0], [0, 0, 1]])),
            base_contour=base_contour_L,
            height=3,
            rotations=[0, 1, 2, 3]
        ),
    }

    global brick_list
    base_brick_list = {}
    for k, v in simple_brick_list.items():
        if v.is_base:
            base_brick_list[k] = v
    all_brick_list = {}

    def update_with_assertion(brick_list):
        '''Assert that all_brick_list doesn't have any bricks in brick_list before being udpated.'''
        assert not set(all_brick_list.keys()) & set(brick_list.keys())
        all_brick_list.update(brick_list)

    update_with_assertion(simple_brick_list)
    update_with_assertion(tile_brick_list)
    update_with_assertion(p5_offset_brick_list)
    if EXCLUDE_LSHAPE:
        print('excluding lshape!!!')
    else:
        update_with_assertion(lshape_brick_list)
    slope_brick_list = get_slope_bricks()
    update_with_assertion(get_slope_bricks())
    misc_brick_list = get_misc_bricks()
    update_with_assertion(misc_brick_list)
    brick_list = {
        # exclusive
        'simple': simple_brick_list,
        'tile': tile_brick_list,
        'p5': p5_offset_brick_list,
        'lshape': lshape_brick_list,
        'slope': slope_brick_list,
        'misc': misc_brick_list,
        #
        'all': all_brick_list,
        'base': base_brick_list,
    }


init_brick_list()


@functools.lru_cache
def get_all_brick_ids():
    return sorted(list(brick_list['all'].keys()))


@functools.lru_cache
def get_brick_ids(brick_type_category):
    return sorted(list(brick_list[brick_type_category].keys()))


@functools.lru_cache
def get_non_simple_brick_ids():
    return list(set(get_brick_ids('all')) - set(get_brick_ids('simple')))


def get_brick_class(bid) -> BasicBrickInfo:
    return brick_list['all'][bid]


def get_brick_info_raw(bid):
    return brick_list['all'][bid].get_info()


def get_brick_info(brick, grid_size, expand_stud=False):
    """

    Args:
        brick:
        grid_size:

    Returns:
        occ_pts, stud_pts, astud_pts, base_contour

    """
    ps = get_brick_info_raw(brick.brick_type)
    ps_transformed = []
    for i in range(len(ps)):
        p = ps[i]
        if p:
            p_round = transform_points_round(p, brick.transform_matrix)
            ps_transformed.append(p_round)
        else:
            ps_transformed.append([])
    # expand occupancy points 2x
    brick_pts = ps_transformed[0]
    brick_pts_exp = set()
    # grid_center = grid_size // 2
    grid_center = np.array([grid_size[0] // 2, grid_size[1] // 8, grid_size[2] // 2])
    xyz_max = grid_size - np.array([1, 1, 1]) - grid_center
    xyz_min = np.array([0.0, 0.0, 0.0]) - grid_center
    if (brick_pts > xyz_max).any() or (brick_pts < xyz_min).any():
        return [None] * 4

    if expand_stud:
        stud_pts = ps_transformed[1]
        stud_pts_exp = set()
        astud_pts = ps_transformed[2]
        astud_pts_exp = set()

    for offset_x in [-0.25, 0.25]:
        for offset_y in [-0.25, 0.25]:
            for offset_z in [-0.25, 0.25]:
                for p in brick_pts:
                    p_offset = p + [offset_x, offset_y, offset_z]
                    brick_pts_exp.add(tuple(p_offset))
                if expand_stud:
                    for p in stud_pts:
                        p_offset = p + [offset_x, offset_y, offset_z]
                        stud_pts_exp.add(tuple(p_offset))
                    for p in astud_pts:
                        p_offset = p + [offset_x, offset_y, offset_z]
                        astud_pts_exp.add(tuple(p_offset))

    if expand_stud:
        ps_transformed = (list(brick_pts_exp), list(stud_pts_exp), list(astud_pts_exp), ps_transformed[3])
    else:
        ps_transformed = (list(brick_pts_exp), ps_transformed[1], ps_transformed[2], ps_transformed[3])
    ps_tuple = []
    for i in range(len(ps_transformed)):
        p = ps_transformed[i]
        if i == 3:
            ps_tuple.append(list(map(tuple, p)))
        else:
            ps_tuple.append(set(map(tuple, p)))
    return ps_tuple


def get_brick_enc_voxel_info():
    stud_id = 2
    bid2id = collections.defaultdict(lambda: 3)
    bid2id.update({
        # round shape
        '3062b': 4,
        '4032a': 4,
        '4073': 4,
        '37352': 4,
        '3941': 4,
        '4150': 4,

        # slope shape (not including all slope bricks)
        '54200': 5,
        '85984': 5,

        # special shape
        '2877': 6,
        '59900': 6,
        '6213': 6,
        '87087': 7,
    })
    id2value = {
        2: 0.25,
        3: 1,
        4: 0.75,
        5: 0.6,
        6: 0.5,
        7: 0.9
    }
    return stud_id, bid2id, id2value


def get_brick_enc_voxel_(bid, position=None, rotation=None, extra_point=None, extra_point_value=None, use_id=False):
    grid_size = [65, 65, 65]
    voxel_size = [130, 130, 130]
    voxel_translation = [-65, -65 // 4, -65]
    stud_id, bid2id, id2value = get_brick_enc_voxel_info()

    transform_mat = None
    if position is not None:
        assert rotation is not None
        transform_mat = tr.concatenate_matrices(tr.translation_matrix(position), tr.quaternion_matrix(rotation))

    brick = Brick(bid, [0, 0, 0], [1, 0, 0, 0])
    occ_pts, studs_pts, astuds_pts, _, = get_brick_info(brick, grid_size, expand_stud=True)

    def pts2voxel(pts, value):
        pts = np.stack(list(pts))
        pts = (pts * 2 - 0.5).round().astype(np.int)
        pts = (pts - voxel_translation).astype(np.int)
        voxel = np.zeros(tuple(voxel_size), dtype=np.float)
        assert not ((pts < 0).any() or (pts > np.array(voxel_size)[None, :]).any())
        voxel[pts[:, 0], pts[:, 1], pts[:, 2]] = value
        return voxel

    if transform_mat is not None:
        occ_pts = transform_points_round(list(occ_pts), transform_mat)
    if use_id:
        occ_voxel = pts2voxel(occ_pts, bid2id[bid])
    else:
        occ_voxel = pts2voxel(occ_pts, id2value[bid2id[bid]])
    if extra_point is not None:
        occ_voxel += pts2voxel([extra_point], extra_point_value)
    if studs_pts:
        if transform_mat is not None:
            studs_pts = transform_points_round(list(studs_pts), transform_mat)
        if use_id:
            stud_voxel = pts2voxel(studs_pts, stud_id)
        else:
            stud_voxel = pts2voxel(studs_pts, id2value[stud_id])
        return occ_voxel + stud_voxel
    else:
        return occ_voxel


brick_enc_voxel_map = {}


def get_brick_enc_voxel(bid, extra_point=None, extra_point_value=2):
    '''
    :param bid: brick type.
    :param extra_point: If extra_point and extra_point_value are set,
    the corresponding position in the voxel will be set to that value
    :return: a voxel representation that is supposed to identify a brick type by
    considering studs, voxel shapes.
    '''
    global brick_enc_voxel_map
    if bid not in brick_enc_voxel_map:
        brick_enc_voxel_map[bid] = get_brick_enc_voxel_(bid,
                                                        extra_point=extra_point, extra_point_value=extra_point_value)
    return brick_enc_voxel_map[bid]


def get_cbrick_enc_voxel(cbrick, extra_point=None, extra_point_value=2):
    voxel = None

    # We let v2 override v1 to deal with those voxels that are originally
    # studs but overridden by occs.
    def merge_voxel(v1, v2):
        v1[v2.nonzero()] = v2[v2.nonzero()]
        return v1

    for i, b in enumerate(cbrick.bricks_raw):
        if voxel is None:
            voxel = get_brick_enc_voxel_(b.brick_type, b.position, b.rotation)
        else:
            if i == len(cbrick.bricks_raw) - 1:
                merge_voxel(voxel,
                            get_brick_enc_voxel_(b.brick_type, b.position, b.rotation, extra_point, extra_point_value))
            else:
                merge_voxel(voxel, get_brick_enc_voxel_(b.brick_type, b.position, b.rotation))

    return voxel


from .utils import chamfer_distance


def get_cbrick_rotations(cbrick):
    '''Check valid rotations by rotation occupancy grid'''
    occ_pts = np.stack(list(cbrick.brick_info_raw[0]))

    rotations = [0]
    occ_pts_rot = np.array(occ_pts)
    for i in range(2):
        occ_pts_rot = transform_points_round(occ_pts_rot, tr.rotation_matrix(np.pi / 2, [0, 1, 0]))
        if chamfer_distance(occ_pts, occ_pts_rot) < 1e-6:
            break
        rotations.append(i + 1)
    # asymmetric
    if len(rotations) == 3:
        rotations.append(3)
    return rotations


def transform_points_round(pts, matrix):
    def round(x):
        return np.round(x * 4) / 4

    p = np.stack(list(pts))
    p = tr.transform_points(p, matrix).tolist()
    p_round = np.vectorize(round)(p)
    if not np.allclose(p, p_round):
        print('Big rounding error in transform_points_round:')
        print('p', p)
        print('p_round', p_round)
        # import ipdb; ipdb.set_trace()
    return p_round


def get_connection_offset(brick_type_or_cbrick, rotation_quat, op_type):
    if isinstance(brick_type_or_cbrick, str):
        _, stud_pts, astud_pts, _ = get_brick_info_raw(brick_type_or_cbrick)
    else:
        assert isinstance(brick_type_or_cbrick, CBrick)
        _, stud_pts, astud_pts, _ = brick_type_or_cbrick.brick_info_raw

    rotation_matrix = tr.quaternion_matrix(rotation_quat)
    if op_type in [0, 2]:
        p = transform_points_round(astud_pts, rotation_matrix)
    else:
        assert op_type == 1
        p = transform_points_round(stud_pts, rotation_matrix)
    return p


# return available stud if the bricks are facing up, else return available astud
def get_brick_valid_positions(bricks, brick_type, rotation, op_type, only_heuristic=False, debug=False):
    offsets = get_connection_offset(brick_type, rotation, op_type)
    res = []
    if debug:
        import ipdb;
        ipdb.set_trace()

    if op_type == 0:
        studs_positions = np.array(list(bricks.get_stud_positions()))
        studs_positions = np.round(tr.transform_points(studs_positions, bricks.transform_.matrix) * 2) / 2
        for offset in offsets:
            for pos in studs_positions:
                res.append(tuple(map(float, pos - offset)))
    else:
        assert op_type == 1
        astuds_positions = np.array(list(bricks.get_astud_positions()))
        astuds_positions = np.round(tr.transform_points(astuds_positions, bricks.transform_.matrix) * 2) / 2
        for offset in offsets:
            for pos in astuds_positions:
                pos_new = pos - offset
                res.append(tuple(map(float, pos_new)))
    res = set(res)

    if only_heuristic:
        return res

    res_filterd = []

    for pos in res:
        if bricks.add_brick(brick_type, pos, rotation, op_type, only_check=True):
            res_filterd.append(pos)

    return set(res_filterd)


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''
    import matplotlib
    matplotlib.use('Agg')

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def visualize_brick_info(t, b, s, a, c):
    import matplotlib
    matplotlib.use('Agg')
    print('base_contour', c)

    def get_coords(l):
        xs, zs, ys = list(zip(*list(l)))
        return xs, ys, zs

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*get_coords(b), marker='o', s=140)
    if s:
        ax.scatter(*get_coords(s), marker='^', s=140)
    if a:
        ax.scatter(*get_coords(a), marker='v', s=140)
    ax.scatter(*get_coords(c), marker='x', s=140)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    set_axes_equal(ax)
    plt.title(t)
    plt.show()


class Brick:
    def __init__(self, brick_type, position, rotation):
        '''
        brick_type: specify the shape of the brick (like 1x1x2, 1x1xl..)
        position
        size:
        # rotation: [0-3] counterclockwise, 0: 0  ; 1: 90  ; 2: 180;  3:270;
        # for L shape, default is ('L' rotated 90 counter clockwise)
        rotation: quaternion
        '''
        self.brick_type = brick_type
        self.position = np.array(position)
        self.rotation = rotation
        self.transform_matrix = \
            tr.concatenate_matrices(tr.translation_matrix(self.position), tr.quaternion_matrix(self.rotation))

    def __repr__(self):
        return f'Brick:{{ type:{self.brick_type},' \
               f' anno:{get_brick_annotation(self.brick_type)},' \
               f' position:{self.position} rotation:{self.rotation} }}'


class VBrick:
    # add only after element brick is added to brick_list

    def __init__(self, n, brick_type, position, rotation):
        """

        Args:
            n: number of bricks to be vertically stacked
            brick_type:
            position:
            rotation:
        """
        self._state_dict = dict(
            cls='VBrick',
            n=n, brick_type=brick_type,
            position=list(map(float, position)),
            rotation=list(map(float, rotation)),
        )

        assert n > 1, (brick_type, position, rotation)  # not necessary
        b_info = get_brick_class(brick_type)
        self.bricks: List[Brick] = []

        for _ in range(n):
            b = Brick(brick_type, position, rotation)
            self.bricks.append(b)

            position, = elevate_pts([position], position[1] + b_info.height)

        self.n = n
        self.height = get_brick_class(brick_type).get_height() * n

    def get_height(self):
        return self.height

    def __repr__(self):
        return f"VBrick:{{ n:{self.n}, " \
               f"brick:{repr(self.bricks[0])}, " \
               f"brick:{repr(self.bricks[-1])}"

    def get_brick_info(self, grid_size):
        ps_tuples = [get_brick_info(b, grid_size) for b in self.bricks]
        occs, studs, astuds, contour = zip(*ps_tuples)

        op_type = get_stack_op_type(self.bricks[0].brick_type)

        # take union of occ_pts
        occ = set().union(*occs)
        assert op_type in [0], print(f'Op type {op_type} not supported for now')
        stud = studs[0]
        astud = astuds[0]
        for stud_cur, astud_cur in zip(studs[1:], astuds[1:]):
            locked_1 = astud_cur & stud
            locked_2 = stud_cur & astud
            if not locked_1 or locked_2:
                import ipdb;
                ipdb.set_trace()
            stud |= stud_cur
            astud |= astud_cur
            stud -= locked_1
            astud -= locked_1

        # take the contour of the bottom brick
        contour = contour[0]

        return occ, stud, astud, contour

    def to_dict(self):
        return self._state_dict


b2_types = []


def init_b2_types():
    global b2_types
    b2_types = []
    for b_type in get_all_brick_ids():
        b_cls = get_brick_class(b_type)
        if b_cls.base_bbox_size[0] != 1 or b_cls.base_bbox_size[1] != 1:
            continue
        b2_types.append(b_type)

    return b2_types


init_b2_types()


def get_b2_types():
    return b2_types


from bricks.utils import line_area


class HBrick:
    # one central brick with dense surrounding bricks
    # this is not a subclass of CBrick because b1, b2s are not glued together
    def __init__(self, b1_brick_type, b1_position, b1_rotation, b2_brick_type='3024'):

        self._state_dict = dict(
            cls='HBrick',
            b1_brick_type=b1_brick_type,
            b1_position=list(map(float, b1_position)),
            b1_rotation=list(map(float, b1_rotation)),
            b2_brick_type=b2_brick_type,
        )

        b1 = Brick(b1_brick_type, b1_position, b1_rotation)

        # compute b2 positions
        b1_info: SimpleBrickInfo = get_brick_class(b1_brick_type)
        b2_info: SimpleBrickInfo = get_brick_class(b2_brick_type)
        b1_size = b1_info.base_bbox_size
        # relative positions to b1 center
        b2_rel_box = [-b1_size[0] / 2 - 0.5, -b1_size[1] / 2 - 0.5, b1_size[0] / 2 + 0.5, b1_size[1] / 2 + 0.5]
        # assume b1 has a rectangular base
        # b2_rel_box = [-b1_info.size[0] / 2 - b2_info.size[0] / 2, -b1_info.size[1] / 2 - b2_info.size[1] / 2,
        #               b1_info.size[0] / 2 + b2_info.size[0] / 2, b1_info.size[1] / 2 + b2_info.size[1] / 2]
        b2_rel_positions = set().union(
            line_area(b2_rel_box[0], b2_rel_box[1], 0, b2_rel_box[0], b2_rel_box[3]),
            line_area(b2_rel_box[0], b2_rel_box[3], 0, b2_rel_box[2], b2_rel_box[3]),
            line_area(b2_rel_box[2], b2_rel_box[3], 0, b2_rel_box[2], b2_rel_box[1]),
            line_area(b2_rel_box[2], b2_rel_box[1], 0, b2_rel_box[0], b2_rel_box[1]),
        )
        b2_positions = offset_pts(b2_rel_positions,
                                  offset_x=b1_position[0], offset_h=b1_position[1],
                                  offset_y=b1_position[2])
        b2_positions = list(b2_positions)
        # don't allow rotation for now
        b2_rotations = [brick_rotation_candidates[0]] * len(b2_positions)

        self.bricks = [b1]
        self.bricks += [Brick(b2_brick_type, b2_positions[i], b2_rotations[i]) for i in range(len(b2_positions))]
        self._check_bricks(self.bricks)

        l = b1_info.base_bbox_size[0] + b2_info.base_bbox_size[0] * 2
        w = b1_info.base_bbox_size[1] + b2_info.base_bbox_size[1] * 2
        base_contour = []
        for sign in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
            base_contour.append(np.array([l / 2 * sign[0], 0, w / 2 * sign[1]]))
        self._eff_base_contour_raw = base_contour

        self.height = max(b1_info.get_height(), b2_info.get_height())

    def get_height(self):
        return self.height

    def get_base_contour_transformed(self):
        b1 = self.bricks[0]
        p = self._eff_base_contour_raw
        p_round = transform_points_round(p, b1.transform_matrix)
        return p_round

    @staticmethod
    def _check_bricks(bricks):
        for b in bricks:
            assert b.position[1] == bricks[0].position[1], [b.position for b in bricks]

    def __repr__(self):
        return f"HBrick:{{ 1+n = 1+{len(self.bricks) - 1}, " \
               f"brick:{repr(self.bricks[0])}, " \
               f"brick:{repr(self.bricks[-1])}"

    def get_brick_info(self, grid_size):
        ps_tuples = [get_brick_info(b, grid_size) for b in self.bricks]
        occ, stud, astud, _ = zip(*ps_tuples)
        occ = set().union(*occ)
        stud = set().union(*stud)
        astud = set().union(*astud)
        contour = self.get_base_contour_transformed()

        return occ, stud, astud, contour

    def to_dict(self):
        return self._state_dict


def _get_stack_op_type(bid):
    b_info = get_brick_class(bid)
    _, stud, astud, _ = get_brick_info_raw(bid)
    stud = set(map(tuple, stud))
    astud = set(map(tuple, astud))

    new_stud = offset_pts(stud, offset_h=b_info.height)
    new_astud = offset_pts(astud, offset_h=b_info.height)

    if stud & new_astud:
        return 0
    if new_stud & astud:
        return 1

    # can't stack
    return -1


bid_to_stack_op_type = {}


def cache_stack_op_type():
    global bid_to_stack_op_type
    for bid in get_all_brick_ids():
        bid_to_stack_op_type[bid] = _get_stack_op_type(bid)


cache_stack_op_type()


def get_stack_op_type(bid):
    return bid_to_stack_op_type[bid]


def get_vbrick_info(brick: Union[Brick, VBrick], grid_size):
    """Should be equivalent to get_brick_info for instances of `Brick`

    Args:
        brick:
        grid_size:

    Returns:

    """
    if isinstance(brick, VBrick):
        return brick.get_brick_info(grid_size)
    else:
        return get_brick_info(brick, grid_size)


def get_hbrick_info(brick: HBrick, grid_size):
    return brick.get_brick_info(grid_size)


import trimesh.transformations as tr
from trimesh.voxel.transforms import Transform


class BricksPC:
    def __init__(self, grid_size=(21, 21, 21), record_parents=True):
        self.bricks = []  # (brick_type, position, rotation)
        self.op_types = []
        self.grid_size = np.array(grid_size)
        self.occupancy = set()
        self.stud = set()
        self.astud = set()
        self.transform_ = Transform(np.eye(4))
        # From grid coordinates to world coordinates
        self.occ_size = 2 * self.grid_size
        # self.occ_translation = np.array(-grid_size)
        self.occ_translation = -np.array([grid_size[0], grid_size[1] // 4, grid_size[2]])

        self.stud_map = {}  # map stud point to brick index
        self.astud_map = {}  # map astud point to brick index
        self.brick_dependency = []  # map each brick to the parent bricks (bricks providing studs)
        self.record_parents = record_parents

        self.base_contours = []

    def apply_transform(self, matrix):
        self.transform_.apply_transform(matrix)

    def get_canonical_pose(self, position, rotation):
        transform_matrix = tr.concatenate_matrices(tr.translation_matrix(position), tr.quaternion_matrix(rotation))
        inverse_matrix = tr.inverse_matrix(self.transform_.matrix)
        transform_matrix = tr.concatenate_matrices(inverse_matrix, transform_matrix)
        position = tr.translation_from_matrix(transform_matrix)
        rotation = tr.quaternion_from_matrix(transform_matrix)
        return position, rotation

    def add_vbrick(self, brick: VBrick, op_type=0, verbose=False, debug=False, only_check=False, vbrick_info=None):
        if debug:
            import ipdb;
            ipdb.set_trace()

        for b in brick.bricks[1:]:
            _, stud, astud, _ = get_brick_info(b, self.grid_size)
            locked_1 = astud & self.stud
            locked_2 = stud & self.astud
            if locked_1 or locked_2:
                print('vbrick top elements connected to previous bricks', locked_1, locked_2)
                return False

        return self.add_brick_(brick, op_type,
                               verbose=verbose or debug, only_check=only_check, brick_info=vbrick_info)

    def add_hbrick(self, brick: HBrick, op_type, verbose=False, debug=False, only_check=False, hbrick_info=None):
        if debug:
            import ipdb;
            ipdb.set_trace()

        # need extra lock check for b2
        # because adding HBrick with add_brick_ assumes that elements are glued together
        for b in brick.bricks[1:]:
            _, stud, astud, _ = get_brick_info(b, self.grid_size)
            locked_1 = astud & self.stud
            locked_2 = stud & self.astud
            if op_type == 0:
                if not locked_1 or locked_2:
                    # if verbose:
                    #     print('b2 type 0 lock check failed', b)
                    return False
            elif op_type == 1:
                if locked_1 or not locked_2:
                    # if verbose:
                    #     print('b2 type 1 lock check failed', b)
                    return False
            else:
                if locked_1 or locked_2:
                    # if verbose:
                    #     print('b2 type 2 lock check failed', b)
                    return False
        return self.add_brick_(brick, op_type, verbose=verbose or debug, only_check=only_check, brick_info=hbrick_info)

    def add_brick(self,
                  brick_type,
                  position=[0, 0, 0],
                  rotation=np.array([1, 0, 0, 0]),
                  op_type=0, canonical=False, verbose=False, debug=False, only_check=False, no_check=False):
        '''
        op_type: 0: new brick's astud -> current bricks' stud
        1: new brick's stud -> current bricks' astud
        2: put on the ground

        For now, ignore the new brick if it has both 0 and 1
        '''
        position = np.array(position)
        rotation = np.array(rotation)  # quaternion
        if debug:
            import traceback
            traceback.print_exc()
            raise ValueError(traceback.format_exc())
            # import ipdb;
            # ipdb.set_trace()
        if not canonical:
            position, rotation = self.get_canonical_pose(position, rotation)
        brick = Brick(brick_type, position, rotation)
        return self.add_brick_(brick, op_type, verbose=verbose or debug, only_check=only_check, no_check=no_check)

    def add_brick_(self, brick: Union[Brick, VBrick, HBrick], op_type, verbose, only_check, brick_info=None,
                   no_check=False):
        # check validity
        # if len(self.bricks) >= 104:
        #     import ipdb; ipdb.set_trace()
        if brick_info is None:
            if isinstance(brick, Brick):
                brick_info = get_brick_info(brick, self.grid_size)
            else:
                brick_info = brick.get_brick_info(self.grid_size)
        occ, stud, astud, base_contour = brick_info

        if no_check:
            if occ is None:
                print('Occ of the brick is None. It\'s probably out of bound.')
                # import ipdb; ipdb.set_trace()
                return False

            self.stud |= stud
            self.astud |= astud
            locked_1 = astud & self.stud
            locked_2 = stud & self.astud
            self.stud = self.stud - locked_1 - locked_2
            self.astud = self.astud - locked_1 - locked_2
            self.occupancy |= occ
            self.bricks.append(brick)
            self.op_types.append(op_type)
            self.base_contours.append(base_contour)
            return True

        if occ is None:
            return False

        def dist(x, y):
            if len(y.shape) == 1:
                return ((x - y) ** 2).sum() ** 0.5
            else:
                assert len(y.shape) == 2
                return ((x[None, :] - y) ** 2).sum(axis=-1) ** 0.5

        if not (occ & self.occupancy):

            # stud/astud collision test

            if len(base_contour) == 0:
                stud_tmp = np.array([])
            else:
                stud_tmp = np.array(list(filter(lambda st: st[1] == base_contour[0][1], self.stud)))

            if len(stud) > 0:
                if len(base_contour) == 0:
                    astud_tmp = np.array([])
                else:
                    if isinstance(brick, Brick):
                        brick_height = get_brick_class(brick.brick_type).get_height()
                    else:
                        brick_height = brick.get_height()
                    astud_tmp = np.array(
                        list(filter(lambda st: st[1] == base_contour[0][1] + brick_height, self.astud)))
                brick_stud_np = np.array(list(stud))
                if astud_tmp.size != 0:
                    dist2 = ((astud_tmp[:, None] - brick_stud_np[None, :]) ** 2).sum(-1)
                    dist2[dist2 == 0] = 1  # ignore connected points
                    if dist2.min() <= 0.25:
                        if verbose:
                            print('Astud collided with brick\'s stud')
                        return False

                # stud/contour collison check
                for c in self.base_contours:
                    for i in range(len(c)):
                        if c[i][1] != brick_stud_np[0][1]:
                            continue
                        j = (i + 1) % len(c)
                        p_i = np.array(c[i])
                        p_j = np.array(c[j])
                        dist_ij = dist(p_i, p_j)
                        dist_i = dist(p_i, np.array(brick_stud_np))
                        dist_j = dist(p_j, np.array(brick_stud_np))

                        # check whether the stud lies in line (p_i, p_j)
                        if (abs(dist_i + dist_j - dist_ij) <= 1e-6).any():
                            if verbose:
                                print('stud/contour check failed')
                            return False

            # contour/stud collision check
            if stud_tmp.size != 0:
                for i in range(len(base_contour)):
                    j = (i + 1) % len(base_contour)
                    p_i = np.array(base_contour[i])
                    p_j = np.array(base_contour[j])
                    dist_ij = dist(p_i, p_j)
                    dist_i = dist(p_i, np.array(stud_tmp))
                    dist_j = dist(p_j, np.array(stud_tmp))

                    # check whether the stud lies in line (p_i, p_j)
                    if (abs(dist_i + dist_j - dist_ij) <= 1e-6).any():
                        if verbose:
                            print('contour/stud check failed')
                        return False

            locked_1 = astud & self.stud
            locked_2 = stud & self.astud
            if op_type == 0:
                if not locked_1 or locked_2:
                    if verbose:
                        print('Lock check failed', locked_1, locked_2, astud, stud)
                    return False
                if not only_check:
                    self.stud |= stud
                    self.astud |= astud
                    self.stud = self.stud - locked_1
                    self.astud = self.astud - locked_1

                    if self.record_parents:
                        parent_bricks = set()
                        for s in locked_1:
                            parent_bricks.add(self.stud_map[s])
                            del self.stud_map[s]
                        self.brick_dependency.append(parent_bricks)

                        for s in stud:
                            self.stud_map[s] = len(self.bricks)
                        for s in astud - locked_1:
                            self.astud_map[s] = len(self.bricks)

            elif op_type == 1:
                if locked_1 or not locked_2:
                    if verbose:
                        print('Lock check failed', locked_1, locked_2)
                    return False
                if not only_check:
                    self.stud |= stud
                    self.astud |= astud
                    self.stud = self.stud - locked_2
                    self.astud = self.astud - locked_2

                    if self.record_parents:
                        for s in locked_2:
                            self.brick_dependency[self.astud_map[s]].add(len(self.bricks))
                            del self.astud_map[s]

                        for s in stud - locked_2:
                            self.stud_map[s] = len(self.bricks)
                        for s in astud:
                            self.astud_map[s] = len(self.bricks)

            else:
                if locked_1 or locked_2:
                    if verbose:
                        print('Lock check failed', locked_1, locked_2)
                    return False
                if not only_check:
                    self.stud |= stud
                    self.astud |= astud

                    if self.record_parents:
                        for s in stud:
                            self.stud_map[s] = len(self.bricks)
                        for s in astud:
                            self.astud_map[s] = len(self.bricks)
                        self.brick_dependency.append(set())

            if not only_check:
                self.occupancy |= occ
                self.bricks.append(brick)
                self.op_types.append(op_type)
                self.base_contours.append(base_contour)
            return True
        if verbose:
            print('Occupancy check failed')
        return False

    def get_stud_positions(self):
        return self.stud

    def get_astud_positions(self):
        return self.astud

    def get_brick_transform(self, ind, rel_ind=None, canonical=False):
        # import ipdb; ipdb.set_trace()
        brick = self.bricks[ind]
        if isinstance(brick, Brick):
            assert rel_ind is None or rel_ind == 0
            return self._get_brick_transform(brick, canonical)
        if rel_ind is not None:
            return self._get_brick_transform(brick.bricks[rel_ind], canonical)
        return [self._get_brick_transform(b, canonical) for b in brick.bricks]

    def _get_brick_transform(self, brick: Brick, canonical):
        position = brick.position
        rotation = brick.rotation
        if not canonical:
            transform_matrix = tr.concatenate_matrices(self.transform_.matrix, brick.transform_matrix)
            position = (np.around(tr.translation_from_matrix(transform_matrix) * 2, decimals=0) / 2).astype(np.float)
            rotation = tr.quaternion_from_matrix(transform_matrix)
            return position, rotation
        return position, rotation

    def get_occ_with_rotation(self):
        '''
        return the occ that are applied with rotation transform
        '''
        rot_occ = np.zeros(self.occ_size, dtype=bool)
        occ_transformed = transform_points_round(list(self.occupancy), self.transform_.matrix)
        occ_pos = (occ_transformed * 2 - 0.5).round().astype(np.int)
        occ_pos -= self.occ_translation[None, :]
        if (occ_pos < 0).any() or (occ_pos > self.occ_size[None, :]).any():
            pass
            # import ipdb; ipdb.set_trace()

        rot_occ[occ_pos[:, 0], occ_pos[:, 1], occ_pos[:, 2]] = True

        # if rot_occ.sum() != len(self.occupancy):
        #     import ipdb; ipdb.set_trace()
        return rot_occ, occ_pos

    def __repr__(self):
        s = 'Bricks {\n'
        for i, b in enumerate(self.bricks):
            s += f' Op type: {self.op_types[i]} '
            s += str(b)
            s += '\n'
        s += '}'
        return s

    def to_dict(self, bricks_nums=None):
        # Bricks can be recovered simply by (op_type, brick)

        d = {}
        if bricks_nums is None:
            bricks_nums = [1] * len(self.bricks)
        bricks_ct = 0
        operations = []
        for bricks_num in bricks_nums:
            bricks_step = []
            for i in range(bricks_ct, bricks_ct + bricks_num):
                b = self.bricks[i]
                b_d = {}
                if isinstance(b, Brick):
                    b_d['brick_type'] = b.brick_type
                    b_d['canonical_position'] = list(map(float, b.position))  # For json serializing
                    b_d['canonical_rotation'] = list(map(float, b.rotation))  # For json serializing
                else:
                    b_d['canonical_state'] = b.to_dict()
                b_d['op_type'] = int(self.op_types[i])
                bricks_step.append(b_d)
            step_d = {
                'bricks': bricks_step
            }
            operations.append(step_d)
            bricks_ct += bricks_num

        d['operations'] = {i: operations[i] for i in range(len(bricks_nums))}
        d['grid_size'] = list(map(int, self.grid_size))  # For json serializing
        return d

    @staticmethod
    def from_dict(d, no_check=False):
        bpc = BricksPC(d['grid_size'])
        for op in d['operations'].values():
            bs = op['bricks']
            for b in bs:
                ret = bpc.add_brick(
                    b['brick_type'],
                    b['canonical_position'],
                    b['canonical_rotation'],
                    b['op_type'], canonical=True, no_check=no_check)
                if not ret:
                    bpc.add_brick(
                        b['brick_type'],
                        b['canonical_position'],
                        b['canonical_rotation'],
                        b['op_type'], canonical=True, debug=True, no_check=no_check)
                    return None
        return bpc


class CBrick:
    # compositional brick

    def __init__(self, bricks_pc: BricksPC, position, rotation):
        # assume that bricks should be added from bricks[0] to bricks[-1]
        # coordinates at origin
        # analogous to brick_info_raw
        # stud = set()
        # astud = bricks_pc.get_astud_positions()
        self.stud_leave_vacant_raw: Set = bricks_pc.get_stud_positions()
        _, _, base_astud, _ = get_brick_info(bricks_pc.bricks[0], bricks_pc.grid_size)
        self.astud_leave_vacant_raw: Set = bricks_pc.get_astud_positions() - base_astud

        self.bricks_raw = bricks_pc.bricks

        # analogous to BrickInfo, but after transformed to bricks_pc.grid_size
        self.brick_info_raw: Tuple[Set, Set, Set, List] = (
            bricks_pc.occupancy, bricks_pc.get_stud_positions(), bricks_pc.get_astud_positions(), []
            # ignore contour for now
        )
        self.grid_size = bricks_pc.grid_size

        # shift all above by position

        self.position = np.array(position)
        self.rotation = rotation
        self.transform_matrix = \
            tr.concatenate_matrices(tr.translation_matrix(self.position), tr.quaternion_matrix(self.rotation))

        self._state_dict = dict(
            cls='CBrick',
            bricks_pc=bricks_pc.to_dict(),
            position=list(map(float, position)),
            rotation=list(map(float, rotation)),
        )

        height = 0
        for b in bricks_pc.bricks:
            height = max(height, get_brick_class(b.brick_type).get_height() + b.position[1])
        self.height = height

    def to_dict(self):
        return self._state_dict

    def get_height(self):
        return self.height

    @property
    def bricks(self):
        # apply transform to bricks
        bricks_transformed = []
        for b in self.bricks_raw:
            # print('before', b)
            assert isinstance(b, Brick), 'not implemented'
            # brick pose in canonical form
            transform_matrix = tr.concatenate_matrices(tr.translation_matrix(b.position),
                                                       tr.quaternion_matrix(b.rotation))

            # apply self.transform_matrix to the brick
            transform_matrix = tr.concatenate_matrices(self.transform_matrix, transform_matrix)
            position = tr.translation_from_matrix(transform_matrix)
            rotation = tr.quaternion_from_matrix(transform_matrix)
            b = Brick(b.brick_type, position, rotation)
            bricks_transformed.append(b)

            # print('after', b)
        return bricks_transformed

    @property
    def stud_leave_vacant(self) -> Set:

        if not self.stud_leave_vacant_raw:
            return set()
        return set(map(tuple, transform_points_round(list(self.stud_leave_vacant_raw), self.transform_matrix)))

    @property
    def astud_leave_vacant(self) -> Set:
        if not self.astud_leave_vacant_raw:
            return set()
        return set(map(tuple, transform_points_round(list(self.astud_leave_vacant_raw), self.transform_matrix)))

    def get_brick_info(self, grid_size):
        ps = self.brick_info_raw

        # the following is adapted from get_brick_info
        ps_transformed = []
        for i in range(len(ps)):
            p = ps[i]
            if p:
                p_round = transform_points_round(list(p), self.transform_matrix)
                ps_transformed.append(p_round)
            else:
                ps_transformed.append([])

        assert list(grid_size) == list(self.grid_size), (grid_size, self.grid_size, 'not implemented')
        ps_tuple = []
        for i in range(len(ps_transformed)):
            p = ps_transformed[i]
            if i == 3:
                ps_tuple.append(list(map(tuple, p)))
            else:
                ps_tuple.append(set(map(tuple, p)))
        return ps_tuple

    def __repr__(self):
        return f'Brick:{{ type: CBrick,' \
               f' position:{self.position} rotation:{self.rotation} }}'


def dict_to_cbrick(b_d, reset_pose=False, no_check=False):
    '''Reset_pose is used when you want a canonical cbrick.'''
    b = BricksPC.from_dict(b_d['canonical_state']['bricks_pc'], no_check=no_check)
    if not b:
        return None
    if not reset_pose:
        return CBrick(b, b_d['brick_transform']['position'], b_d['brick_transform']['rotation'])
    else:
        return CBrick(b, [0, 0, 0], [1, 0, 0, 0])


def cbrick_highest_brick(cbrick: CBrick, return_ind=False, allow_multiple=False):
    max_ind = None
    max_height = None
    for i, b in enumerate(cbrick.bricks_raw):
        this_height = b.position[1] + get_brick_class(b.brick_type).get_height()
        if max_ind is None:
            if allow_multiple:
                max_ind = [i]
            else:
                max_ind = i
            max_height = this_height
        else:
            if this_height > max_height:
                if allow_multiple:
                    max_ind = [i]
                else:
                    max_ind = i
                max_height = this_height
            elif this_height == max_height and allow_multiple:
                max_ind.append(i)

    if return_ind or allow_multiple:
        if allow_multiple:
            return [cbrick.bricks[i] for i in max_ind], max_ind
        else:
            return cbrick.bricks[max_ind], max_ind
    else:
        return cbrick.bricks[max_ind]


def get_cbrick_keypoint(cbrick: CBrick, policy='simple', top_ind=-1):
    '''
    :param policy: if 'simple',  uses origin + [0, height, 0] as the origin. If 'brick', select a brick with the
        greatest height and use it's keypoint as the keypoint of the cbrick.
    :return: a 3d np.array if 'simple'. an extra index pointing to the brick
    '''
    if policy == 'simple':
        return np.array(cbrick.position) + [0, cbrick.get_height(), 0]
    else:
        assert policy == 'brick', print("Unknown keypoint policy.")
        if top_ind < 0:
            brick, max_ind = cbrick_highest_brick(cbrick, return_ind=True)
        else:
            bricks, max_inds = cbrick_highest_brick(cbrick, return_ind=True, allow_multiple=True)
            brick = bricks[top_ind]
            max_ind = max_inds[top_ind]
        return np.array(brick.position) + [0, get_brick_class(brick.brick_type).get_height(), 0], max_ind


def add_cbrick_to_bricks_pc(bricks_pc: BricksPC, brick: CBrick, op_type=0, verbose=False, debug=False, only_check=False,
                            cbrick_info=None,
                            no_check=False):
    locked_1 = brick.astud_leave_vacant & bricks_pc.stud
    locked_2 = brick.stud_leave_vacant & bricks_pc.astud

    return bricks_pc.add_brick_(
        brick, op_type,
        verbose=verbose or debug, only_check=only_check, brick_info=cbrick_info, no_check=no_check)

# 1_1_1 3024
# 1_1_2 3005
# 1_2_1 3023
# 1_2_2 3004
# 1_3_1 3623
# 1_3_2 3622
# 1_4_2 3010
# 1_6_1 3666
# 1_6_2 3009
# 1_8_1 3460
# 1_8_2 3008
# 2_2_1 3022
# 2_2_2 3003
# 2_3_1 3021
# 2_4_1 3020
# 2_4_2 3001
# 2_6_1 3795
# 2_6_2 2456
# 2_8_1 3034
# 2_8_2 3007
# 6_8_1 3036
# 1_1_f 3070
# 1_4_f 2431
# 3_L_1 2420
# 3_L_2 2357
# 2x2 center: 87580
# 1x2 center: 3794a
