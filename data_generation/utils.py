from typing import List, Union

import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
from PIL import ImageDraw, Image
from pytorch3d.renderer import (
    look_at_view_transform,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    BlendParams,
)
from pytorch3d.structures import join_meshes_as_scene


def round_corner(radius, fill):
    """Draw a round corner"""
    corner = Image.new('RGBA', (radius, radius), (0, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=fill)
    return corner


def round_rectangle(size, radius, fill):
    """Draw a rounded rectangle"""
    width, height = size
    rectangle = Image.new('RGBA', size, fill)
    corner = round_corner(radius, fill)
    rectangle.paste(corner, (0, 0))
    rectangle.paste(corner.rotate(90), (0, height - radius))  # Rotate the corner and paste it
    rectangle.paste(corner.rotate(180), (width - radius, height - radius))
    rectangle.paste(corner.rotate(270), (width - radius, 0))
    return rectangle


class SimpleShader(torch.nn.Module):
    def __init__(self, blend_params, device="cpu"):
        super().__init__()
        self.blend_params = blend_params

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        pixel_colors = meshes.sample_textures(fragments)
        images = pytorch3d.renderer.blending.hard_rgb_blend(pixel_colors, fragments, self.blend_params)
        return images  # (N, H, W, 3) RGBA image


import pycocotools._mask as _mask


def encode(bimask):
    if len(bimask.shape) == 3:
        return _mask.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return _mask.encode(bimask.reshape((h, w, 1), order='F'))[0]


def decode(rleObjs):
    if type(rleObjs) == list:
        return _mask.decode(rleObjs)
    else:
        return _mask.decode([rleObjs])[:, :, 0]


def area(rleObjs):
    if type(rleObjs) == list:
        return _mask.area(rleObjs)
    else:
        return _mask.area([rleObjs])[0]


def toBbox(rleObjs):
    if type(rleObjs) == list:
        return _mask.toBbox(rleObjs)
    else:
        return _mask.toBbox([rleObjs])[0]


def mask2bbox(mask):
    pos = np.argwhere(mask)
    y0 = int(pos[:, 0].min())
    x0 = int(pos[:, 1].min())
    y1 = int(pos[:, 0].max())
    x1 = int(pos[:, 1].max())
    return [(x0, y0), (x1, y1)]


def transform_mesh(mesh, transform):
    verts = mesh.verts_padded()
    verts_transform = transform.transform_points(verts)
    return mesh.update_padded(verts_transform)


from bricks.bricks import get_brick_canonical_mesh
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from torch import nn
import pytorch3d.transforms as pt
import trimesh.transformations as tr
import time


def brick2p3dmesh(brick, color=None, cuda=False):
    mesh = get_brick_canonical_mesh(brick.brick_type)
    verts_tensor = torch.Tensor(mesh.vertices[None, :, :])
    faces_tensor = torch.Tensor(mesh.faces[None, :, :])
    textures = None
    if color is not None:
        verts_rgb = torch.zeros(mesh.vertices.shape).unsqueeze(dim=0)
        verts_rgb[..., :] = torch.Tensor(np.array(color) / 255)
        textures = TexturesVertex(verts_features=verts_rgb)
    position = np.array(brick.position)
    scale_d = {'x': 20, 'y': 8, 'z': 20}
    trans = [position[0] * scale_d['x'], position[1] * scale_d['y'], position[2] * scale_d['z']]
    R = torch.as_tensor(tr.quaternion_matrix(brick.rotation)[:3, :3].T)
    transform = pt.Transform3d().compose(pt.Rotate(R)).compose(pt.Translate(*trans))
    mesh = Meshes(verts=verts_tensor, faces=faces_tensor, textures=textures)
    if cuda:
        transform = transform.cuda()
        mesh = mesh.cuda()
    mesh = transform_mesh(mesh, transform)
    return mesh


from bricks.brick_info import Brick, CBrick


def bricks2meshes(bricks: List[Union[Brick, CBrick]], colors):
    meshes = []
    for i, b in enumerate(bricks):
        if isinstance(b, CBrick):
            meshes_cbrick = []
            for j in range(len(b.bricks_raw)):
                if isinstance(colors[i][0], list):
                    color = colors[i][j]
                else:
                    color = colors[i]
                meshes_cbrick.append(brick2p3dmesh(b.bricks[j], color).cuda())
            meshes.append(join_meshes_as_scene(meshes_cbrick))
        else:
            meshes.append(brick2p3dmesh(b, colors[i]).cuda())
    return meshes


class MeshRendererWithDepth(nn.Module):
    def __init__(self, rasterizer, shader):
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        z_buf = fragments.zbuf
        min_v = z_buf[z_buf > -1].min()
        max_v = z_buf[z_buf > -1].max()
        bg_idxs = z_buf == -1
        z_buf = (z_buf - min_v) / (max_v - min_v)
        z_buf[bg_idxs] = 0
        # z_buf += 1
        # z_buf[z_buf > 0] = 1
        return images, z_buf


def render_lego_scene(mesh, cameras):
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=1e-8,
        faces_per_pixel=1,
        max_faces_per_bin=100000
    )

    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(1, 1, 1))

    renderer = MeshRendererWithDepth(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings,
            cameras=cameras
        ),

        shader=HardPhongShader(
            device=mesh.device,
            blend_params=blend_params
        )
    )

    lights_location = np.array([0, 1000000, 0])
    lights = PointLights(
        # ambient_color=((0.5, 0.5, 0.5),),
        # diffuse_color=((0.5, 0.5, 0.5),),
        # specular_color=((0.0, 0.0, 0.0),),
        device=mesh.device, location=(lights_location,))

    image, depth_image = renderer(mesh, cameras=cameras, lights=lights)
    return image, depth_image


def get_brick_masks(bricks_mesh_list, mask_colors, idxs, cameras, render_size=1536, output_size=512):
    raster_settings_simple = RasterizationSettings(
        image_size=render_size,
        blur_radius=1e-8,
        max_faces_per_bin=100000
    )

    mesh_all = join_meshes_as_scene(bricks_mesh_list).cuda()

    simple_blend_params = BlendParams(sigma=0, gamma=0, background_color=(0, 0, 0))
    renderer_simple = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings_simple,
            cameras=cameras
        ),

        shader=SimpleShader(simple_blend_params)
    )
    # [1, H, W, 3]
    image_shadeless = renderer_simple(mesh_all, cameras=cameras)

    brick_masks = []
    for k in idxs:
        ref_pixel = torch.Tensor(mask_colors[k]).reshape(1, 1, 1, -1).to('cuda:0')
        brick_mask = ((image_shadeless[0, :, :, :3] * 255).round() == ref_pixel).all(dim=-1)
        brick_mask = F.interpolate(brick_mask[None,].float(), size=output_size)[0].bool()
        brick_masks.append(brick_mask[0].cpu().numpy())

    return brick_masks, image_shadeless


from lego.utils.data_generation_utils import flatten_nested_list, map_nested_list, index_list


def get_brick_masks_for_nested(bricks_mesh_list, mask_colors, idxs, cameras, render_size=1536, output_size=512):
    raster_settings_simple = RasterizationSettings(
        image_size=render_size,
        blur_radius=1e-8,
        max_faces_per_bin=100000
    )

    mesh_all = join_meshes_as_scene(flatten_nested_list(bricks_mesh_list, template=bricks_mesh_list)).cuda()

    simple_blend_params = BlendParams(sigma=0, gamma=0, background_color=(0, 0, 0))
    renderer_simple = MeshRenderer(
        rasterizer=MeshRasterizer(
            raster_settings=raster_settings_simple,
            cameras=cameras
        ),

        shader=SimpleShader(simple_blend_params)
    )
    # [1, H, W, 3]
    image_shadeless = renderer_simple(mesh_all, cameras=cameras)

    def fn(mask_color):
        ref_pixel = torch.FloatTensor(mask_color).view(1, 1, 1, 3).cuda()
        brick_mask = ((image_shadeless[:, :, :, :3] * 255).round() == ref_pixel).all(dim=-1).squeeze(0)
        # -> (h, w)
        brick_mask = F.interpolate(
            brick_mask.view(1, 1, render_size, render_size).float(),
            size=output_size).view(output_size, output_size).bool()
        # -> (h, w)

        # to numpy
        return brick_mask.cpu().numpy()

    mask_colors_selected = index_list(mask_colors, idxs, check=False)
    list_of_brick_mask = map_nested_list(fn, mask_colors_selected,
                                         template=index_list(bricks_mesh_list, idxs, check=False))
    return list_of_brick_mask, image_shadeless


def highlight_edge(image_pil, depth_map, image_shadeless):
    import cv2
    med_size = 1024
    depth_map_np = (depth_map[0, :, :, 0].detach().cpu().numpy() * 255).astype(np.float32)
    img_shadeless_np = (image_shadeless[0, :, :, :3] * 255).detach().cpu().numpy().astype(np.float32)
    img_shadeless_lap = np.array(cv2.Laplacian(img_shadeless_np, cv2.CV_32F))
    img_shadeless_lap = (img_shadeless_lap.max(axis=-1)).astype(np.float32) * 255

    # img_shadeless_lap = np.array(cv2.Canny(img_shadeless_np.astype(np.uint8), 255 * 0.3, 255))
    # import ipdb; ipdb.set_trace()
    depth_map_lap = np.array(cv2.Laplacian(depth_map_np, cv2.CV_32F))
    depth_map_pil = Image.fromarray(depth_map_np.astype(np.uint8))
    depth_map_lap_pil = Image.fromarray(depth_map_lap.astype(np.uint8))
    img_shadeless_lap_pil = Image.fromarray(img_shadeless_lap.astype(np.uint8))
    # img_shadeless_lap_pil = img_shadeless_lap_pil.filter(ImageFilter.MinFilter(3))
    img_shadeless_lap_pil = img_shadeless_lap_pil.resize((med_size, med_size))
    depth_map_pil = depth_map_pil.resize((med_size, med_size))
    depth_map_lap_pil = depth_map_lap_pil.resize((med_size, med_size))
    merged_lap_np = np.maximum(np.array(img_shadeless_lap_pil) >= 1, np.array(depth_map_lap_pil) >= 1).astype(
        np.uint8) * 255
    merged_lap = Image.fromarray(255 - merged_lap_np)

    image_pil.paste(merged_lap.resize((512, 512)), (0, 0), Image.fromarray(merged_lap_np).resize((512, 512)))

    return image_pil


from contextlib import contextmanager


@contextmanager
def named_timeit(name, store_dict) -> float:
    if name not in store_dict:
        store_dict[name] = 0
    t = time.perf_counter()
    yield
    store_dict[name] += time.perf_counter() - t
