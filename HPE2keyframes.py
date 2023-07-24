import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation   
from scipy.ndimage import binary_erosion, binary_dilation

import os
import json



def euler_angles_smooth(XYZ_euler_angles):

    if XYZ_euler_angles.ndim == 1:
        XYZ_euler_angles = XYZ_euler_angles[:, np.newaxis]

    for i in range(XYZ_euler_angles.shape[0]-1):
        for j in range(XYZ_euler_angles.shape[1]):
            # smooth
            if XYZ_euler_angles[i+1, j] - XYZ_euler_angles[i, j] > 180:
                XYZ_euler_angles[i+1:, j] = XYZ_euler_angles[i+1:, j] - 360
            elif XYZ_euler_angles[i+1, j] - XYZ_euler_angles[i, j] < -180:
                XYZ_euler_angles[i+1:, j] = XYZ_euler_angles[i+1:, j] + 360

    return np.squeeze(XYZ_euler_angles)



def xyz2euler_body(xyz, xyz_body_frame, X_dir=1.0, Y_dir=1.0):
    '''
        xyz: Coordinates from 3D human pose estimation. Each dimension: (frame, 3, xyz)
        xyz_body_frame: Coordinates of body frame. Used to calculate the Y direction rotation of body.
        X_dir: -1.0 for arm and body.
        Y_dir: -1.0 for body and head.
    '''

    # swap y and z to align the coordinate in the mine-imator
    xyz[:, :, [1, 2]] = xyz[:, :, [2, 1]]
    xyz[:, :, 0] = -xyz[:, :, 0]
    xyz_body_frame[:, :, [1, 2]] = xyz_body_frame[:, :, [2, 1]]
    xyz_body_frame[:, :, 0] = -xyz_body_frame[:, :, 0]

    p0, p1, p2 = torch.unbind(xyz, dim=1)
    p1_, p4_, p14_, p11_ = torch.unbind(xyz_body_frame, dim=1)

    # solve the cosine pose matrix
    Y = (p0 - p1) * Y_dir
    arm = p2 - p1

    Y = F.normalize(Y, dim=1)
    X = F.normalize(p11_ + p4_ - p1_ - p14_, dim=1)
    # X = F.normalize(torch.cross(X_dir*arm, Y), dim=1)  # TODO smooth
    Z = F.normalize(torch.cross(X, Y), dim=1)

    cos_pose_matrix = torch.stack([X, Y, Z], dim=2)
    r =  Rotation.from_matrix(cos_pose_matrix)
    YXZ_euler_angles = r.as_euler("YXZ", degrees=True)

    # bend
    bend = -(Y * F.normalize(arm, dim=1)).sum(dim=1) * Y_dir
    bend = torch.rad2deg(torch.acos(bend))

    # swap xyz
    YXZ_euler_angles[:, [0, 1, 2]] = YXZ_euler_angles[:, [1, 0, 2]]
    XYZ_euler_angles = YXZ_euler_angles

    # arm cos_pose_matrix
    Y_arm = F.normalize(arm, dim=1)
    X_arm = X
    Z_arm = F.normalize(torch.cross(X_arm, Y_arm), dim=1)
    cos_pose_matrix_arm = torch.stack([X_arm, Y_arm, Z_arm], dim=2)

    # avoid abrupt changes in angle
    XYZ_euler_angles = euler_angles_smooth(XYZ_euler_angles)
    bend = euler_angles_smooth(bend.numpy())

    return XYZ_euler_angles, bend, cos_pose_matrix_arm


def xyz2euler_relative(xyz, cos_body, X_dir=1.0, Y_dir=1.0, head=False, leg=False, euler_body=None):
    '''
        xyz: Coordinates from 3D human pose estimation. Each dimension: (frame, 3, xyz)
        X_dir: -1.0 for arm and body.
        Y_dir: -1.0 for body and head.
    '''

    # swap y and z to align the coordinate in the mine-imator
    xyz[:, :, [1, 2]] = xyz[:, :, [2, 1]]
    xyz[:, :, 0] = -xyz[:, :, 0]
    p0, p1, p2 = torch.unbind(xyz, dim=1)

    # solve the cosine pose matrix
    Y = (p0 - p1) * Y_dir
    arm = p2 - p1

    Y = F.normalize(Y, dim=1)
    X = F.normalize(torch.cross(X_dir*arm, Y), dim=1)  # TODO smooth
    Z = F.normalize(torch.cross(X, Y), dim=1)

    cos_pose_matrix = torch.stack([X, Y, Z], dim=2)

    if head == True:
        Y_arm = F.normalize(arm, dim=1)
        X_arm = X
        Z_arm = F.normalize(torch.cross(X_arm, Y_arm), dim=1)
        cos_pose_matrix = torch.stack([X_arm, Y_arm, Z_arm], dim=2)

    # relative to the body rotation Y
    if leg == True:
        euler_body_Y = euler_body * 0
        euler_body_Y[:, 0:1] = euler_body[:, 1:2]
        r_body_Y = Rotation.from_euler("YXZ", euler_body_Y, degrees=True)
        cos_body_Y = torch.from_numpy(r_body_Y.as_matrix())

    # relative to the body
    cos_relative = cos_body if leg == False else cos_body_Y.float()
    cos_pose_matrix = cos_relative.permute(0, 2, 1) @ cos_pose_matrix
    r =  Rotation.from_matrix(cos_pose_matrix)
    YXZ_euler_angles = r.as_euler("YXZ", degrees=True)

    # bend
    bend = -(Y * F.normalize(arm, dim=1)).sum(dim=1) * Y_dir
    bend = torch.rad2deg(torch.acos(bend))
    # if head == True:
    #     bend = bend * 0.5

    # swap xyz
    YXZ_euler_angles[:, [0, 1, 2]] = YXZ_euler_angles[:, [1, 0, 2]]
    XYZ_euler_angles = YXZ_euler_angles

    # avoid abrupt changes in angle
    XYZ_euler_angles = euler_angles_smooth(XYZ_euler_angles)
    bend = euler_angles_smooth(bend.numpy())

    return XYZ_euler_angles, bend


def calculate_body_offset(euler_body, euler_right_leg, bend_right_leg, euler_left_leg, bend_left_leg, length_leg=[6, 6], prior=False):
    '''
        Calculate the offset of the body to make the movement more realistic. 
        First, determine the foot positions of both legs based on the actual 
        effect of Euler angle rotation in Mine-imator. Then, determine which 
        leg is currently touching the ground and fix the grounded leg. This 
        allows the calculation of the body offset.

    '''

    def calculate_leg_coordinates(r_body_Y, euler_leg, bend_leg, length_leg, right=True):
        YXZ_euler_leg = euler_leg[:, [1, 0, 2]]
        r1 = Rotation.from_euler("YXZ", YXZ_euler_leg, degrees=True)
        m1 = r1.as_matrix()
        X1 = m1[:, :, 0]  # direction
        Y1 = m1[:, :, 1]  # vector to be rotated
        r2 = Rotation.from_rotvec(X1*bend_leg[:, np.newaxis], degrees=True)
        Y2 = r2.apply(Y1)  # reconstruct the arm vector
        coordinates = -(Y1 * length_leg[0] + Y2 * length_leg[1])
        coordinates[:, 0] = coordinates[:, 0] - 2 
        coordinates = r_body_Y.apply(coordinates)
        return coordinates

    # calculate the endpoint coordinates of two legs
    euler_body_Y = euler_body * 0
    euler_body_Y[:, 0:1] = euler_body[:, 1:2]
    r_body_Y = Rotation.from_euler("YXZ", euler_body_Y, degrees=True)
    right_coordinates = calculate_leg_coordinates(r_body_Y, euler_right_leg, bend_right_leg, length_leg)
    left_coordinates = calculate_leg_coordinates(r_body_Y, euler_left_leg, bend_left_leg, length_leg)
    # stack, 0: right, 1: left
    coordinates = np.stack([right_coordinates, left_coordinates], axis=1)
    
    # determine which leg grounded, 0: right, 1: left
    grounded_flag = (right_coordinates[:, 1] > left_coordinates[:, 1])*1
    # prior knowledge: The more bended legs are not grounded
    if prior == True:
        grounded_flag_left = (bend_right_leg - bend_left_leg) > 30
        grounded_flag_right = (bend_left_leg - bend_right_leg) > 30
        grounded_flag += grounded_flag_left*1
        grounded_flag *= (1 - grounded_flag_right)*1
    # smoothing
    grounded_flag = binary_erosion(grounded_flag, structure=np.ones(7))*1
    grounded_flag = binary_dilation(grounded_flag, structure=np.ones(7))*1

    body_POS = np.zeros_like(right_coordinates)

    # POS_Y
    ind = np.array(range(right_coordinates.shape[0]))
    body_POS[:, 1] = -coordinates[ind, grounded_flag, 1]

    # extract the X, Z coordinates of grounded leg in time t_1 
    X_t1 = coordinates[ind[:-1], grounded_flag[:-1], 0]
    Z_t1 = coordinates[ind[:-1], grounded_flag[:-1], 2]
    # extract the X, Z coordinates of grounded leg in time t_2
    # note that the split of grounded_flag not changed
    X_t2 = coordinates[ind[1:], grounded_flag[:-1], 0]
    Z_t2 = coordinates[ind[1:], grounded_flag[:-1], 2]

    # calculate the relative displacement between two frames
    X_relative = X_t2 - X_t1
    Z_relative = Z_t2 - Z_t1

    # calculate the absolute displacement
    X_abs = np.cumsum(X_relative)
    Z_abs = np.cumsum(Z_relative)

    body_POS[1:, 0] = -X_abs
    body_POS[1:, 2] = -Z_abs

    return body_POS


def add_keyframes(data, length, part_name, euler, bend, not_body=True, not_head=True, body_steve=False, body_POS=None):
    for i in range(length):
        if not_head:
            keyframes_dict = {
                "position": i,
                "part_name": part_name,
                "values": {
                    "ROT_X": float(euler[i][0]),
                    "ROT_Y": float(euler[i][2]),  # Y, Z args in mine-imator miframes is exchanged. Maybe a bug. 
                    "ROT_Z": float(euler[i][1]*not_body),
                    "BEND_ANGLE_X": float(bend[i])
                }
            }
        else:  # no bend
            keyframes_dict = {
                "position": i,
                "part_name": part_name,
                "values": {
                    "ROT_X": float(euler[i][0]),
                    "ROT_Y": float(euler[i][2]),
                    "ROT_Z": float(euler[i][1]),
                }
            }
        if body_steve == True:
            keyframes_dict = {
                "position": i,
                "values": {
                    "POS_X": float(body_POS[i][0]),
                    "POS_Y": float(body_POS[i][2]),
                    "POS_Z": float(body_POS[i][1]),
                    "ROT_Z": float(euler[i][1])
			    }
            }
        data["keyframes"].append(keyframes_dict)

    print(f"add_key_frames: {part_name}")


def hpe2keyframes(HPE_filename, FPS_mine_imator, keyframes_filename, prior=True):

    # read data
    with open(HPE_filename, 'rb') as file:
        data = np.load(file)
    print(f"open file: {HPE_filename}")
    xyz = data.copy()
    length = xyz.shape[0]

    # extract data from each body part
    xyz_right_leg = torch.from_numpy(xyz[:, 1:4, :])
    xyz_right_arm = torch.from_numpy(xyz[:, 14:17, :])
    xyz_left_leg = torch.from_numpy(xyz[:, 4:7, :])
    xyz_left_arm = torch.from_numpy(xyz[:, 11:14, :])
    xyz_body = torch.from_numpy(xyz[:, [0, 7, 8], :])
    xyz_body_frame = torch.from_numpy(xyz[:, [1, 4, 14, 11], :])
    xyz_head = torch.from_numpy(xyz[:, [8, 9, 10], :])

    # calculate the absolute euler angles of body
    euler_body, bend_body, cos_pos_matrix = xyz2euler_body(xyz_body, xyz_body_frame, X_dir=-1, Y_dir=-1)

    # calculate the relative euler angles of arm and head with respect to the body ROT_Y
    euler_right_leg, bend_right_leg = xyz2euler_relative(xyz_right_leg, cos_pos_matrix, leg=True, euler_body=euler_body)
    euler_left_leg, bend_left_leg = xyz2euler_relative(xyz_left_leg, cos_pos_matrix, leg=True, euler_body=euler_body)
    
    # calculate the relative euler angles of arm and head with respect to the upper body
    euler_right_arm, bend_right_arm = xyz2euler_relative(xyz_right_arm, cos_pos_matrix, X_dir=-1)
    euler_left_arm, bend_left_arm = xyz2euler_relative(xyz_left_arm, cos_pos_matrix, X_dir=-1)
    euler_head, bend_head = xyz2euler_relative(xyz_head, cos_pos_matrix, Y_dir=-1, head=True)
    
    # create json format data
    data = {
        "format": 34,
        "created_in": "2.0.0",  # mine-imator version
        "is_model": True,
        "tempo": FPS_mine_imator,  # FPS
        "length": length,  # keyframes length
        "keyframes": [
        ],
        "templates": [],
        "timelines": [],
        "resources": []
    }
    
    # relative offset makes the model more realistic
    # caculate the relative offset based on Euler angle and bending angle
    body_POS = calculate_body_offset(euler_body, euler_right_leg, bend_right_leg, euler_left_leg, bend_left_leg, prior=prior)


    add_keyframes(data, length, "left_leg", euler_left_leg, bend_left_leg)
    add_keyframes(data, length, "right_leg", euler_right_leg, bend_right_leg)
    add_keyframes(data, length, "left_arm", euler_left_arm, bend_left_arm)
    add_keyframes(data, length, "right_arm", euler_right_arm, bend_right_arm)
    add_keyframes(data, length, "body", euler_body, bend_body, not_body=False)
    add_keyframes(data, length, "head", euler_head, bend_head, not_head=False)
    add_keyframes(data, length, "abc", euler_body, bend_body, body_steve=True, body_POS=body_POS)  # TODO

    # save json
    with open(keyframes_filename, "w") as file:
        json.dump(data, file, indent=4)

    print(f"keyframes file saves successfully, file path: {os.path.abspath(keyframes_filename)}")
    


if __name__ == '__main__':
    # config
    HPE_filename = "outputs/test_3d_output_malaoshi_2-00_2-30_postprocess.npy"
    FPS_mine_imator = 30
    keyframes_filename = "steve_malaoshi2.miframes"
    prior = True
    hpe2keyframes(HPE_filename, FPS_mine_imator, keyframes_filename, prior=prior)

print("Done!")
