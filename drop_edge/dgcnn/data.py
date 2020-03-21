#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from model import knn
import torch

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def rotate_pointcloud(pointcloud):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data = np.dot(pointcloud, rotation_matrix).astype('float32')
    return rotated_data

def shift_point_cloud(pointcloud, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, shifted batch of point clouds
    """
    shifts = np.random.uniform(-shift_range, shift_range, size=[3])
    pointcloud += shifts
    return pointcloud.astype('float32')


def random_scale_point_cloud(pointcloud, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, scaled batch of point clouds
    """
    scales = np.random.uniform(scale_low, scale_high, size=[1])
    pointcloud *= scales
    return pointcloud.astype('float32')

def rotate_perturbation_point_cloud(pointcloud, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
      Input:
        BxNx3 array, original batch of point clouds
      Return:
        BxNx3 array, rotated batch of point clouds
    """
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
             [0,np.cos(angles[0]),-np.sin(angles[0])],
             [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
             [0,1,0],
             [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
             [np.sin(angles[2]),np.cos(angles[2]),0],
             [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    rotated_data = np.dot(pointcloud, R).astype('float32')
    return rotated_data

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            #pointcloud = translate_pointcloud(pointcloud)
            pointcloud = rotate_pointcloud(pointcloud)
            pointcloud = jitter_pointcloud(pointcloud).astype('float32')
            pointcloud = random_scale_point_cloud(pointcloud)
            pointcloud = rotate_perturbation_point_cloud(pointcloud)
            pointcloud = shift_point_cloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    '''train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)'''
        
    B = 9840
    N = 1024
    K = 20
    
    data,_ = load_data('train')
    print('data type:', type(data))
    print('data shape:', data.shape)
    data = data[:B,:N,:]
    print('data shape:', data.shape)   #(10,1024,3)
    data = torch.from_numpy(data)
    data = data.view(B, -1, N)
    
    ne_idx = knn(data, K)
    print('ne_idx shape:',ne_idx.shape)
    #print('ne_idx:\n', ne_idx)
    ne_idx = ne_idx.reshape(B*N*K)
    print('ne_idx:', ne_idx)
    
    #device = torch.device('cuda')
    #center_idx = torch.arange(0, N, device=device).view(1, N, 1).repeat(B, 1, K).reshape(B*N*K)
    center_idx = torch.arange(0, N).view(1, N, 1).repeat(B, 1, K).reshape(B*N*K)
    print('center_idx shape:', center_idx.shape)
    print('center_idx:\n', center_idx)
    
    batch_idx = torch.arange(0, B).view(B,1,1).repeat(1,N,K).reshape(B*N*K)
    print('batch_idx shape:', batch_idx.shape)
    print('batch_idx:\n', batch_idx)    
    
    A = torch.zeros(N,N).view(1,N,N).repeat(B,1,1)
    A[batch_idx,center_idx,ne_idx]=1
    print('A.shape:', A.shape)
    print('A[0]', A[0])
    print('sum(A[0]):', torch.sum(A[0]))
    
    
