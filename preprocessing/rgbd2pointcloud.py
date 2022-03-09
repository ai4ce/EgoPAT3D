# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 21:14:03 2021

@author: LENOVO
"""
import open3d as o3d
import os
import numpy as np
import re

class FPSsampling:
    def __init__(self, points):
        self.points = points #np.unique(points, axis=0)
 
    def calculate_distance(self, a, b):
        distance = []
        for i in range(a.shape[0]):
            dis = np.sum(np.square(a[i] - b), axis=-1)
            distance.append(dis)
        distance = np.stack(distance, axis=-1)
        distance = np.min(distance, axis=-1)
        return np.argmax(distance)
 
    def generatesample(self, K):
        firstpoint=np.random.choice(np.arange(0,self.points.shape[0]))

        A = self.points[firstpoint].reshape(1,3)
        B = np.delete(self.points, firstpoint, 0)
        indexlist = [firstpoint]
        for i in range(1,K):
            max_id = self.calculate_distance(A, B)
            A = np.append(A, np.array([B[max_id]]), 0)
            B = np.delete(B, max_id, 0)
            indexlist.append(max_id)
        return A,indexlist
        
def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point
    
def generatesingle(colorpath,depathpath,savepath,num):

    color_raw = o3d.io.read_image(colorpath)
    depth_raw = o3d.io.read_image(depathpath)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)
    
    inter = o3d.camera.PinholeCameraIntrinsic()
    inter.set_intrinsics(3840, 2160, 1.80820276e+03, 1.80794556e+03, 1.94228662e+03, 1.12382178e+03)
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, inter)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    point=pcd.voxel_down_sample(0.02)

    aa=0.005
    while len(point.points)<8192:
        point=pcd.voxel_down_sample(0.02-aa)
        aa=aa+0.005

    o3d.io.write_point_cloud(os.path.join(savepath,str(1+num)+'.ply'),point)
    


path=os.getcwd()
videolist=[]
for count in os.listdir(path):
    try:
        a=int(count[-1])
        videolist.append(count)
    except:
       None
       
def tryint(s):                    
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):            
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
videolist.sort(key=str2int)

for eachvideo in videolist:
    file=os.listdir(os.path.join(path,eachvideo))
    if 'pointcloud' in file:
        continue
    else:
        if os.path.exists(os.path.join(path,eachvideo,'pointcloud'))==0:
            os.mkdir(os.path.join(path,eachvideo,'pointcloud'))

        colorbasepath=os.path.join(path,eachvideo,'color')
        debasepath=os.path.join(path,eachvideo,'d2rgb')
        for num in range(len(os.listdir(debasepath))):
            
            colorpath=os.path.join(colorbasepath,str(num+1)+'.jpg')
            depathpath=os.path.join(debasepath,str(num+1)+'.png')
            
            generatesingle(colorpath,depathpath,os.path.join(path,eachvideo,'pointcloud'),num)
        
        
        
        


