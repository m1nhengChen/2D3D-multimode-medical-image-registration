from torch import nn
import torch
import ProSTGrid
import numpy as np
import gc
import math

PI = math.pi


def inv_pose_vec(transform_mat, pt):
    #  pt: B * 8 * 3 (X, Y, Z)
    # vec: B * 6 (Rx, Ry, Rz, Tx, Ty, Tz)
    pt = pt - transform_mat[:, :3, 3].unsqueeze(1).repeat(1, 8, 1)
    rot_Mat = transform_mat[:, :3, :3]
    inv_rotMat = torch.inverse(rot_Mat)
    inv_pt = pt.bmm(inv_rotMat)

    return inv_pt


def raydist_range(transform_mat, pt, src):
    inv_pt = inv_pose_vec(transform_mat, pt)
    # inv_pt = inv_pt.squeeze(0)
    inv_pt[:, :, 2] = src - inv_pt[:, :, 2]
    inv_pt = inv_pt.view(-1, 3)
    dist_pt = torch.sqrt(
        torch.mul(inv_pt[:, 0], inv_pt[:, 0]) + torch.mul(inv_pt[:, 1], inv_pt[:, 1]) + torch.mul(inv_pt[:, 2],
                                                                                                  inv_pt[:, 2]))
    dist_min = torch.min(dist_pt)
    dist_max = torch.max(dist_pt)
    return dist_min, dist_max


def _repeat(x, n_repeats):
    with torch.no_grad():
        rep = torch.ones((1, n_repeats), dtype=torch.float32).cuda()

    return torch.matmul(x.view(-1, 1), rep).view(-1)


def _bilinear_interpolate_no_torch_5D(vol, grid):
    # Assume CT to be Nx1xDxHxW
    num_batch, channels, depth, height, width = vol.shape
    vol = vol.permute(0, 2, 3, 4, 1)
    _, out_depth, out_height, out_width, _ = grid.shape
    x = width * (grid[:, :, :, :, 0] * 0.5 + 0.5)
    y = height * (grid[:, :, :, :, 1] * 0.5 + 0.5)
    z = depth * (grid[:, :, :, :, 2] * 0.5 + 0.5)

    x = x.view(-1)
    y = y.view(-1)
    z = z.view(-1)

    ind = ~((x >= 0) * (x <= width) * (y >= 0) * (y <= height) * (z >= 0) * (z <= depth))
    # do sampling
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1
    z0 = torch.floor(z)
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, width - 1)
    x1 = torch.clamp(x1, 0, width - 1)
    y0 = torch.clamp(y0, 0, height - 1)
    y1 = torch.clamp(y1, 0, height - 1)
    z0 = torch.clamp(z0, 0, depth - 1)
    z1 = torch.clamp(z1, 0, depth - 1)

    dim3 = float(width)
    dim2 = float(width * height)
    dim1 = float(depth * width * height)
    dim1_out = float(out_depth * out_width * out_height)

    base = _repeat(torch.arange(start=0, end=num_batch, dtype=torch.float32).cuda() * dim1, np.int32(dim1_out))
    idx_a = base.long() + (z0 * dim2).long() + (y0 * dim3).long() + x0.long()
    idx_b = base.long() + (z0 * dim2).long() + (y0 * dim3).long() + x1.long()
    idx_c = base.long() + (z0 * dim2).long() + (y1 * dim3).long() + x0.long()
    idx_d = base.long() + (z0 * dim2).long() + (y1 * dim3).long() + x1.long()
    idx_e = base.long() + (z1 * dim2).long() + (y0 * dim3).long() + x0.long()
    idx_f = base.long() + (z1 * dim2).long() + (y0 * dim3).long() + x1.long()
    idx_g = base.long() + (z1 * dim2).long() + (y1 * dim3).long() + x0.long()
    idx_h = base.long() + (z1 * dim2).long() + (y1 * dim3).long() + x1.long()

    # use indices to lookup pixels in the flat image and keep channels dim
    im_flat = vol.contiguous().view(-1, channels)
    Ia = im_flat[idx_a].view(-1, channels)
    Ib = im_flat[idx_b].view(-1, channels)
    Ic = im_flat[idx_c].view(-1, channels)
    Id = im_flat[idx_d].view(-1, channels)
    Ie = im_flat[idx_e].view(-1, channels)
    If = im_flat[idx_f].view(-1, channels)
    Ig = im_flat[idx_g].view(-1, channels)
    Ih = im_flat[idx_h].view(-1, channels)

    wa = torch.mul(torch.mul(x1 - x, y1 - y), z1 - z).view(-1, 1)
    wb = torch.mul(torch.mul(x - x0, y1 - y), z1 - z).view(-1, 1)
    wc = torch.mul(torch.mul(x1 - x, y - y0), z1 - z).view(-1, 1)
    wd = torch.mul(torch.mul(x - x0, y - y0), z1 - z).view(-1, 1)
    we = torch.mul(torch.mul(x1 - x, y1 - y), z - z0).view(-1, 1)
    wf = torch.mul(torch.mul(x - x0, y1 - y), z - z0).view(-1, 1)
    wg = torch.mul(torch.mul(x1 - x, y - y0), z - z0).view(-1, 1)
    wh = torch.mul(torch.mul(x - x0, y - y0), z - z0).view(-1, 1)

    interpolated_vol = torch.mul(wa, Ia) + torch.mul(wb, Ib) + torch.mul(wc, Ic) + torch.mul(wd, Id) \
                       + torch.mul(we, Ie) + torch.mul(wf, If) + torch.mul(wg, Ig) + torch.mul(wh, Ih)
    interpolated_vol[ind] = 0.0
    interpolated_vol = interpolated_vol.view(num_batch, out_depth, out_height, out_width, channels)
    interpolated_vol = interpolated_vol.permute(0, 4, 1, 2, 3)

    return interpolated_vol


def set_matrix(BATCH_SIZE, device, proj_parameters):
    radian_x = proj_parameters[:, 0]
    radian_y = proj_parameters[:, 1]
    radian_z = proj_parameters[:, 2]
    x_mov = proj_parameters[:, 3]
    y_mov = proj_parameters[:, 4]
    z_mov = proj_parameters[:, 5]
    rotation_x = torch.cat(
        (torch.tensor([[1, 0, 0, 0]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1),
         torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.cos(radian_x).unsqueeze(1).unsqueeze(1),
                    -torch.sin(radian_x).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device)), 2),
         torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.sin(radian_x).unsqueeze(1).unsqueeze(1),
                    torch.cos(radian_x).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device)), 2),
         torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1)), 1)
    rotation_y = torch.cat(
        (torch.cat((torch.cos(radian_y).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.sin(radian_y).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device)), 2),
         torch.tensor([[0, 1, 0, 0]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1),

         torch.cat((-torch.sin(radian_y).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.cos(radian_y).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device)), 2),
         torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1)), 1)
    rotation_z = torch.cat(
        (torch.cat((torch.cos(radian_z).unsqueeze(1).unsqueeze(1),
                    -torch.sin(radian_z).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device)), 2),
         torch.cat((torch.sin(radian_z).unsqueeze(1).unsqueeze(1),
                    torch.cos(radian_z).unsqueeze(1).unsqueeze(1),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device)), 2),
         torch.tensor([[0, 0, 1, 0]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1),
         torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1)), 1)
    trans_mat = torch.cat(
        (torch.cat((torch.ones((BATCH_SIZE, 1, 1), dtype=torch.float,
                               requires_grad=True, device=device),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    x_mov.unsqueeze(1).unsqueeze(1)), 2),
         torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.ones((BATCH_SIZE, 1, 1), dtype=torch.float,
                               requires_grad=True, device=device),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    y_mov.unsqueeze(1).unsqueeze(1)), 2),
         torch.cat((torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.zeros((BATCH_SIZE, 1, 1), dtype=torch.float,
                                requires_grad=True, device=device),
                    torch.ones((BATCH_SIZE, 1, 1), dtype=torch.float,
                               requires_grad=True, device=device),
                    z_mov.unsqueeze(1).unsqueeze(1)), 2),
         torch.tensor([[0, 0, 0, 1]], dtype=torch.float, requires_grad=True,
                      device=device).repeat(BATCH_SIZE, 1, 1)), 1)
    rot_mat = rotation_z.bmm(rotation_y).bmm(rotation_x)
    transform_mat3x4 = torch.bmm(rot_mat, trans_mat)[:, :3, :]
    return transform_mat3x4


class ProST(nn.Module):
    def __init__(self):
        super(ProST, self).__init__()

    def forward(self, ct, fixed, pose, corner_pt, param, H, W):
        src = param[0]
        det = param[1]
        pix_spacing = param[2]
        step_size = param[3]

        transform_mat3x4 = set_matrix(1, 'cuda', pose)
        dist_min, dist_max = raydist_range(transform_mat3x4, corner_pt, src)
        grid = ProSTGrid.forward(corner_pt, fixed.size(), dist_min.data, dist_max.data,
                                 src, det, pix_spacing, step_size, False)
        grid_trans = grid.bmm(transform_mat3x4.transpose(1, 2)).view(1, H, W, -1, 3)
        x_3d = _bilinear_interpolate_no_torch_5D(ct, grid_trans)
        moving = torch.sum(x_3d, dim=-1)

        return moving


def prost_generator(pose, param, device, norm_factor, ct, x_ray, corner_pt, H, W):
    with torch.no_grad():
        gc.collect()
        torch.cuda.empty_cache()
        rx, ry, rz, tx, ty, tz = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3], pose[:, 4], pose[:, 5]
        start_value = torch.zeros((1, 6)).to(device)
        start_value[:, 0] = torch.tensor(rx)
        start_value[:, 1] = torch.tensor(rz)
        start_value[:, 2] = torch.tensor(ry)
        start_value[:, 3] = torch.tensor(-tz)
        start_value[:, 4] = torch.tensor(-ty)
        start_value[:, 5] = torch.tensor(tx)
        start_value[:, :3] = start_value[:, :3] / 180 * PI
        start_value[:, 3:] = start_value[:, 3:] / norm_factor
        pose = start_value.clone().detach().requires_grad_(False)
        projmodel = ProST(param).to(device)
        target = projmodel(ct, x_ray, pose, corner_pt, param, H, W)
    return target
