# -*- coding: utf-8 -*-
"""
Created on 2021/6/7

@author: he
"""


from data_prepare.SegFix_offset_helper import DTOffsetHelper
import numpy as np
import torch


def circshift(matrix_ori, direction, shiftnum1, shiftnum2):
    # direction = 1,2,3,4 # 偏移方向 1:左上; 2:右上; 3:左下; 4:右下;
    c, h, w = matrix_ori.shape
    matrix_new = np.zeros_like(matrix_ori)

    for k in range(c):
        matrix = matrix_ori[k]
        # matrix = matrix_ori[:,:,k]
        if (direction == 1):
            # 左上
            matrix = np.vstack((matrix[shiftnum1:, :], np.zeros_like(matrix[:shiftnum1, :])))
            matrix = np.hstack((matrix[:, shiftnum2:], np.zeros_like(matrix[:, :shiftnum2])))
        elif (direction == 2):
            # 右上
            matrix = np.vstack((matrix[shiftnum1:, :], np.zeros_like(matrix[:shiftnum1, :])))
            matrix = np.hstack((np.zeros_like(matrix[:, (w - shiftnum2):]), matrix[:, :(w - shiftnum2)]))
        elif (direction == 3):
            # 左下
            matrix = np.vstack((np.zeros_like(matrix[(h - shiftnum1):, :]), matrix[:(h - shiftnum1), :]))
            matrix = np.hstack((matrix[:, shiftnum2:], np.zeros_like(matrix[:, :shiftnum2])))
        elif (direction == 4):
            # 右下
            matrix = np.vstack((np.zeros_like(matrix[(h - shiftnum1):, :]), matrix[:(h - shiftnum1), :]))
            matrix = np.hstack((np.zeros_like(matrix[:, (w - shiftnum2):]), matrix[:, :(w - shiftnum2)]))
        # matrix_new[k]==>matrix_new[:,:, k]
        # matrix_new[:,:, k] = matrix
        matrix_new[k] = matrix

    return matrix_new

def generate_dd_map(label_direction, direction_classes):
    direction_offsets = DTOffsetHelper.label_to_vector(torch.from_numpy(label_direction.reshape(1, label_direction.shape[0], label_direction.shape[1])),direction_classes)
    direction_offsets = direction_offsets[0].permute(1,2,0).detach().cpu().numpy()

    direction_os = direction_offsets #[256,256,2]

    height, weight = direction_os.shape[0], direction_os.shape[1]

    cos_sim_map = np.zeros((height, weight), dtype=np.float)

    feature_list = []
    feature5 = direction_os  # .transpose(1, 2, 0)
    if (direction_classes - 1 == 4):
        direction_os = direction_os.transpose(2, 0, 1)
        feature2 = circshift(direction_os, 1, 1, 0).transpose(1, 2, 0)
        feature4 = circshift(direction_os, 3, 0, 1).transpose(1, 2, 0)
        feature6 = circshift(direction_os, 4, 0, 1).transpose(1, 2, 0)
        feature8 = circshift(direction_os, 3, 1, 0).transpose(1, 2, 0)

        feature_list.append(feature2)
        feature_list.append(feature4)
        # feature_list.append(feature5)
        feature_list.append(feature6)
        feature_list.append(feature8)

    elif (direction_classes - 1 == 8 or direction_classes - 1 == 16):
        direction_os = direction_os.transpose(2, 0, 1) # [2,256,256]
        feature1 = circshift(direction_os, 1, 1, 1).transpose(1, 2, 0)
        feature2 = circshift(direction_os, 1, 1, 0).transpose(1, 2, 0)
        feature3 = circshift(direction_os, 2, 1, 1).transpose(1, 2, 0)
        feature4 = circshift(direction_os, 3, 0, 1).transpose(1, 2, 0)
        feature6 = circshift(direction_os, 4, 0, 1).transpose(1, 2, 0)
        feature7 = circshift(direction_os, 3, 1, 1).transpose(1, 2, 0)
        feature8 = circshift(direction_os, 3, 1, 0).transpose(1, 2, 0)
        feature9 = circshift(direction_os, 4, 1, 1).transpose(1, 2, 0)

        feature_list.append(feature1)
        feature_list.append(feature2)
        feature_list.append(feature3)
        feature_list.append(feature4)
        # feature_list.append(feature5)
        feature_list.append(feature6)
        feature_list.append(feature7)
        feature_list.append(feature8)
        feature_list.append(feature9)

    cos_value = np.zeros((height, weight, direction_classes - 1), dtype=np.float32)
    # print('cos_value.shape = {}'.format(cos_value.shape))
    for k, feature_item in enumerate(feature_list):
        fenzi = (feature5[:, :, 0] * feature_item[:, :, 0] + feature5[:, :, 1] * feature_item[:, :, 1])
        fenmu = (np.sqrt(pow(feature5[:, :, 0], 2) + pow(feature5[:, :, 1], 2)) * np.sqrt(
            pow(feature_item[:, :, 0], 2) + pow(feature_item[:, :, 1], 2)) + 0.000001)
        cos_np = fenzi / fenmu
        cos_value[:, :, k] = cos_np

    cos_value_min = np.min(cos_value, axis=2)
    cos_sim_map = cos_value_min
    cos_sim_map[label_direction == 0] = 1

    cos_sim_map_np = (1 - np.around(cos_sim_map))
    cos_sim_map_np_max = np.max(cos_sim_map_np)
    cos_sim_map_np_min = np.min(cos_sim_map_np)
    cos_sim_map_np_normal = (cos_sim_map_np - cos_sim_map_np_min) / (cos_sim_map_np_max - cos_sim_map_np_min)

    return cos_sim_map_np_normal




