import cv2
import numpy as np
from loss import NCCS, NCCL
from module import prost_generator
import torch
import os

loss_func = NCCL
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device_ids = [0]
device = torch.device('cuda:{}'.format(device_ids[0]))


def gaussian(ori_image, down_times=2):
    # 1：add the original image
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        temp_gau = cv2.pyrDown(temp_gau)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid


def calculate_loss(pose, ct, fixed, corner_pt, param, norm_factor, H, W):
    moving = prost_generator(pose, param, device, norm_factor, ct, fixed, corner_pt, H, W)
    return loss_func(moving, fixed)


# Golden Section method for linear search
def line_search_test(a, b, f0, xtol, image_fixed, ct, corner_pt, param, norm_factor, H, W):
    a1 = a + 0.382 * (b - a)
    a2 = a + 0.618 * (b - a)
    f1, f2 = calculate_loss(a1, image_fixed, ct, corner_pt, param, norm_factor, H, W).cpu(), calculate_loss(a2,
                                                                                                            image_fixed,
                                                                                                            ct,
                                                                                                            corner_pt,
                                                                                                            param,
                                                                                                            norm_factor,
                                                                                                            H, W).cpu()
    while np.dot((b - a).squeeze(), (b - a).squeeze()) > xtol * xtol:
        if f1 < f2:
            if f1 < f0:
                return np.around(a1, 1)
            else:
                b, a2, f2 = a2, a1, f1
                a1 = b - 0.618 * (b - a)
                f1 = calculate_loss(a1, image_fixed, ct, corner_pt, param, norm_factor, H, W).cpu()
        else:
            if f2 < f0:
                return np.around(a2, 1)
            else:
                a, a1, f1 = a1, a2, f2
                a2 = a + 0.618 * (b - a)
                f2 = calculate_loss(a2, image_fixed, ct, corner_pt, param, norm_factor, H, W).cpu()
    a = (a + b) / 2
    a = np.around(a, 1)
    return a


def get_line_bounds(start_parm, direction, down_bound, up_bound):
    # start_parm=np.array([-1,-0.5]),direction=np.array([2,4]),down_bound=np.array([-1,-1]),up_bound=np.array([1,1])
    bound = [[], []]

    flag = 0
    for i in range(len(direction)):

        if abs(direction[i]) < 1e-5:
            continue
        # print(down_bound,start_parm)
        k1 = (down_bound[i] - start_parm[0][i]) / direction[i]
        #         print("k1=",k1)
        if check_valid(start_parm, direction, down_bound, up_bound, k1):
            bound[flag] = (start_parm + k1 * direction).tolist()
            #             print(flag," ",bound[flag])
            flag = flag + 1
            if flag == 2:
                res = np.array(bound)
                return res[0], res[1]

        k2 = (up_bound[i] - start_parm[0][i]) / direction[i]
        #         print("k2=",k2)
        if check_valid(start_parm, direction, down_bound, up_bound, k2):
            bound[flag] = (start_parm + k2 * direction).tolist()
            #             print(flag," ",bound[flag])
            flag = flag + 1
            if flag == 2:
                res = np.array(bound)
                return res[0], res[1]

    print("error in get line bounds,Start point is in bound")
    res = np.array(bound)
    print("bounds:", res[0], res[1])
    print("start_parm", start_parm)
    print("down_bound", down_bound)
    print("up_bound", up_bound)
    print("direction", direction)
    return res[0], res[1]


def check_valid(start_parm, direction, down_bound, up_bound, k):
    err = 1e-5
    point = start_parm + k * direction
    for i in range(len(point)):
        if point[0][i] > up_bound[i] + err or point[0][i] < down_bound[i] - err:
            return False
    return True


def powell_search_double(down_bound, up_bound, start_parm, tol, xtol, image_fixed,
                         max_iterations, corner_pt, param, ct, norm_factor, H, W):
    directions = np.identity(6)
    pos = start_parm
    f_val_ori = calculate_loss(pos, ct, image_fixed, corner_pt, param, norm_factor, H, W).cpu()
    f_val = f_val_ori
    iteration_count = 0
    order = [3, 1, 2, 4, 5, 0]
    # store all the intermediate results
    global list_err
    while True:
        iteration_count = iteration_count + 1
        f_val_pre = f_val
        pos_pre = pos
        list_one_iteration = []
        lambda_turn = []
        list_err = []

        for j in range(6):
            i = order[j]
            direction = directions[i].squeeze()
            a, b = get_line_bounds(pos, direction, down_bound, up_bound)
            if len(a) == 0 or len(b) == 0:
                print("get line bound error")
                return f_val, pos
            pos_pre_line = pos
            pos = line_search_test(a, b, f_val, xtol, image_fixed, ct, corner_pt, param,
                                   norm_factor, H, W)
            if (pos < pos_pre_line).all():
                up_bound = pos_pre_line
            if (pos > pos_pre_line).all():
                down_bound = pos_pre_line
            f_val_pre_line = f_val
            f_val = calculate_loss(pos, ct, image_fixed, corner_pt, param, norm_factor, H, W).cpu()
            if (f_val > f_val_pre_line):
                f_val = f_val_pre_line
                pos = pos_pre_line
            lambda_turn.append(np.dot((np.array(pos_pre_line) - np.array(pos)).squeeze(),
                                      (np.array(pos_pre_line) - np.array(pos)).squeeze()))
            list_one_iteration.append(f_val)

        new_direction = (pos - pos_pre).squeeze()
        if np.dot(new_direction, new_direction) < 1e-5:
            print("early stop")
            return f_val, pos

        a, b = get_line_bounds(pos, new_direction, down_bound, up_bound)
        if len(a) == 0 or len(b) == 0:
            print("get line bound error")
            return f_val, pos
        pos_pre = pos
        pos = line_search_test(a, b, f_val, xtol, image_fixed, ct, corner_pt, param,
                               norm_factor, H, W)
        if (pos < pos_pre).all():
            up_bound = pos_pre_line
        if (pos > pos_pre).all():
            down_bound = pos_pre_line
        f_val = calculate_loss(pos, ct, image_fixed, corner_pt, param, norm_factor, H, W).cpu()
        list_one_iteration.append(f_val)
        list_err.append(list_one_iteration)
        max_lambda = 0.0
        res_directions = np.array(directions, copy=True)
        for i in range(6):
            if lambda_turn[i] > max_lambda:
                t = np.array(directions, copy=True)
                t[i, :] = new_direction
                if abs(np.linalg.det(t)) > 1e-5:
                    res_directions = t
                    max_lambda = lambda_turn[i]
        if abs(max_lambda) < 1e-5:
            print("Error updating direction vector")
            return f_val, pos
        directions = res_directions
        if (abs(f_val_pre - f_val) < tol) or (iteration_count >= max_iterations):
            print("------------------------------------\nEnd of search")
            if (iteration_count == max_iterations):
                print("Maximum number of iterations reached")
            break

    return f_val, pos


def multi_resolution_search(ct, x_ray, initial_pose, corner_pt, param, norm_factor, H, W):
    gaussian_pyramid = gaussian(x_ray, down_times=2)
    lower_bound_1 = np.array([90, 0, 0, 900, 0, 0], dtype=np.float32) + np.array([-40, -40, -40, -100, -40, -25],
                                                                                 dtype=np.float32)
    upper_bound_1 = np.array([90, 0, 0, 900, 0, 0], dtype=np.float32) + np.array([40, 40, 40, 200, 40, 25],
                                                                                 dtype=np.float32)
    _, pos = powell_search_double(lower_bound_1, upper_bound_1, initial_pose, 1, 0.1, gaussian_pyramid[2],
                                  max_iterations=10,
                                  corner_pt=corner_pt, param=param, norm_factor=norm_factor, H=H / 4, W=W / 4)
    lower_bound_2 = np.array([90, 0, 0, 900, 0, 0], dtype=np.float32) + np.array([-20, -20, -20, -50, -20, -10],
                                                                                 dtype=np.float32)
    upper_bound_2 = np.array([90, 0, 0, 900, 0, 0], dtype=np.float32) + np.array([20, 20, 20, 50, 20, 10],
                                                                                 dtype=np.float32)
    _, pos = powell_search_double(lower_bound_2, upper_bound_2, pos, 0.1, 0.01, gaussian_pyramid[1], max_iterations=5,
                                  corner_pt=corner_pt, param=param, norm_factor=norm_factor, H=H / 2, W=W / 2)
    lower_bound_3 = np.array([90, 0, 0, 900, 0, 0], dtype=np.float32) + np.array([-2, -2, -2, -5, -5, -5],
                                                                                 dtype=np.float32)
    upper_bound_3 = np.array([90, 0, 0, 900, 0, 0], dtype=np.float32) + np.array([2, 2, 2, 5, 5, 5],
                                                                                 dtype=np.float32)
    _, pos = powell_search_double(lower_bound_3, upper_bound_3, pos, 1e-3, 1e-4, gaussian_pyramid[0], max_iterations=5,
                                  corner_pt=corner_pt, param=param, norm_factor=norm_factor, H=H, W=W)
    return pos
