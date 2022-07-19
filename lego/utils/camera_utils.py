import pytorch3d.renderer.cameras as prc
import torch
from pytorch3d.renderer import FoVOrthographicCameras

scale = torch.as_tensor([20, 8, 20])
scale_xyz = [0.0024, 0.0024, 0.0024]


def get_scale(device='cuda'):
    return scale.to(device)


def get_cameras(azim, elev, device='cuda'):
    R, T = prc.look_at_view_transform(dist=2000, elev=elev, azim=azim, at=((0, 0, 0),),
                                      device=device)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=[scale_xyz])
    return cameras
