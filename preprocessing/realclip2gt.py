import open3d as o3d
import numpy as np
from functools import reduce
import cv2
import re

import os

def calculate_odometry(sourcepath,idx):
# RGB-D odometry
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    pinhole_camera_intrinsic.set_intrinsics(3840, 2160, 1.80820276e+03, 1.80794556e+03, 1.94228662e+03, 1.12382178e+03)
    

    source_color_path = os.path.join(sourcepath,'color',str(idx+1)+'.jpg')
    source_depth_path = os.path.join(sourcepath,'d2rgb',str(idx+1)+'.png') 
    target_color_path = os.path.join(sourcepath,'color',str(idx)+'.jpg')
    target_depth_path = os.path.join(sourcepath,'d2rgb',str(idx)+'.png') 
    
    source_color = o3d.io.read_image(source_color_path)
    source_depth = o3d.io.read_image(source_depth_path)
    target_color = o3d.io.read_image(target_color_path)
    target_depth = o3d.io.read_image(target_depth_path)
    
    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
   
    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)

    
    [success_hybrid_term, trans_hybrid_term, info] = \
        o3d.pipelines.odometry.compute_rgbd_odometry(source_rgbd_image, \
                                                           target_rgbd_image, pinhole_camera_intrinsic,\
                                                               odo_init,\
                                                                   o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                                                                                                
    return np.array(trans_hybrid_term),success_hybrid_term
                                         





def transform_hand_end_to_start(start_idx, end_idx, x_hand_end, y_hand_end, z_hand_end,sourcepath):

    # for visualization of point cloud
    color_raw_end = o3d.io.read_image(os.path.join(sourcepath,'color', str(end_idx) + ".jpg"))
    depth_raw_end = o3d.io.read_image(os.path.join(sourcepath,'d2rgb', str(end_idx) + ".png"))
    rgbd_image_end = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_end, depth_raw_end)

    color_raw_start = o3d.io.read_image(os.path.join(sourcepath,'color', str(start_idx) + ".jpg"))
    depth_raw_start = o3d.io.read_image(os.path.join(sourcepath,'d2rgb', str(start_idx) + ".png"))
    rgbd_image_start = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw_start, depth_raw_start)
    
    # calculate transformation matrix of end2start
    ######################
    odometry_list=[]
    if os.path.exists(os.path.join(sourcepath,'transformation'))==0:
        os.mkdir(os.path.join(sourcepath,'transformation'))
    if os.path.exists(os.path.join(sourcepath,'transformation','odometry'))==0:
        os.mkdir(os.path.join(sourcepath,'transformation','odometry'))
    if os.path.exists(os.path.join(sourcepath,'transformation','success'))==0:
        os.mkdir(os.path.join(sourcepath,'transformation','success'))
    for num in range(end_idx-start_idx):
        
        try:
            eaodometry=np.load(os.path.join(sourcepath,'transformation','odometry',str(start_idx+num)+'.npy')) 
            succ=np.load(os.path.join(sourcepath,'transformation','success',str(start_idx+num)+'.npy'))  
            odometry_list.append(eaodometry)
        except:
            eaodometry,succ=calculate_odometry(sourcepath,start_idx+num)
            np.save(os.path.join(sourcepath,'transformation','odometry',str(start_idx+num)+'.npy'), eaodometry) 
            np.save(os.path.join(sourcepath,'transformation','success',str(start_idx+num)+'.npy'), succ)  
            odometry_list.append(eaodometry)
    
    odometry=reduce(np.dot, odometry_list)
    
    
    
    ######################
    hand_start = odometry.dot(np.array([x_hand_end, y_hand_end, z_hand_end, 1]).T)

    if cv2.imread(os.path.join(sourcepath,'check',str(end_idx) + "end.jpg")) is None: 
        
       
    # transform the hand position to the first frame
        
       
        
        # for visualization of hand 
        hand_end_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[x_hand_end, y_hand_end, z_hand_end])
        hand_start_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[hand_start[0], hand_start[1], hand_start[2]])
    
    
        inter = o3d.camera.PinholeCameraIntrinsic()
        inter.set_intrinsics(3840, 2160, 1.80820276e+03, 1.80794556e+03, 1.94228662e+03, 1.12382178e+03)
        pcd_start = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_start, inter)
        pcd_end = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image_end, inter)
        
        # Flip it, otherwise the pointcloud will be upside down
        pcd_start.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        hand_start_vis.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    
        pcd_end.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        hand_end_vis.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_start)
        vis.update_geometry(pcd_start)
        vis.poll_events()
        vis.update_renderer()
        vis.add_geometry(hand_start_vis)
        vis.update_geometry(hand_start_vis)
        vis.poll_events()
        vis.update_renderer()
        if os.path.exists(os.path.join(sourcepath,'check'))==0:
            os.mkdir(os.path.join(sourcepath,'check'))
    
        vis.capture_screen_image(os.path.join(sourcepath,'check',str(start_idx) + "start.jpg"))
        vis.destroy_window()
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd_end)
        vis.update_geometry(pcd_end)
        vis.poll_events()
        vis.update_renderer()
        vis.add_geometry(hand_end_vis)
        vis.update_geometry(hand_end_vis)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(os.path.join(sourcepath,'check',str(end_idx) + "end.jpg"))
        vis.destroy_window()
    
    return hand_start[0], hand_start[1], hand_start[2]


def locate_hand_in_3d(u_hand, v_hand, idx,sourcepath):

    # (u_hand, v_hand): hand center in the image coordinate
    # idx: index of the last frame in a sequence
    
    u = int(u_hand*3840)
    v = int(v_hand*2160)
    
    # get the depth value of the hand center
    depth_raw = o3d.io.read_image(os.path.join(sourcepath,'d2rgb',str(idx) + ".png"))
    depth_value = np.asarray(depth_raw)
    d = depth_value[v,u]
    
    # transform the point from image coordinate to 3d coordinate
    z = d/1000
    x = (u-1.94228662e+03)*z/1.80820276e+03
    y = (v-1.12382178e+03)*z/1.80794556e+03
    
    return x,y,z

basepath=os.getcwd()
videolist=[]

for count in os.listdir(basepath):
        try:
            a=int(count[-1])
            videolist.append(count)
        except:
           None
           
if os.path.exists(os.path.join(basepath,'groundtruth'))==0:
    os.mkdir(os.path.join(basepath,'groundtruth'))

def tryint(s):                    
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):            
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
videolist.sort(key=str2int)

clippa=os.listdir(os.path.join(basepath,'newclips'))
clippa.sort(key=str2int)
for video in videolist:
    try:
        ff=open(os.path.join(basepath,'groundtruth',video+'.txt'),'r')
        ff.close()
        continue
    except:
    
        cliptxtpath=os.path.join(basepath,'newclips',video+'.txt')
        videopath=os.path.join(basepath,video)
        f1=open(os.path.join(basepath,'groundtruth',video+'.txt'),'w')
        f=open(cliptxtpath,'r')
        alldata=f.readlines()
        for line in alldata:
            data=line.strip('\n').split(',')
            if len(data)==7:
                start,mid,end=data[0],data[1],data[2]
                midx,midy,endx,endy=data[3],data[4],data[5],data[6]
                midhandxyz=locate_hand_in_3d(1-float(midx),float(midy),int(mid),videopath)
                endhandxyz=locate_hand_in_3d(1-float(endx),float(endy),int(end),videopath)
                
                midhandx,midhandy,midhandz=transform_hand_end_to_start(int(start),int(mid),midhandxyz[0],midhandxyz[1],midhandxyz[2],videopath)
                endhandx,endhandy,endhandz=transform_hand_end_to_start(int(mid),int(end),endhandxyz[0],endhandxyz[1],endhandxyz[2],videopath)
                
                newdata=data[0]+','+data[1]+','+data[2]\
                    +','+str(midhandx)+','+str(midhandy)+','+str(midhandz)\
                        +','+str(endhandx)+','+str(endhandy)+','+str(endhandz)+'\n'
                        
                f1.write(newdata)
                
            elif len(data)==4:
                start,mid=data[0],data[1]
                midx,midy=data[2],data[3]
                midhandxyz=locate_hand_in_3d(1-float(midx),float(midy),mid,videopath)
                
                midhandx,midhandy,midhandz=transform_hand_end_to_start(int(start),int(mid),midhandxyz[0],midhandxyz[1],midhandxyz[2],videopath)
                
                newdata=data[0]+','+data[1]\
                    +','+str(midhandx)+','+str(midhandy)+','+str(midhandz)+'\n'
                        
                f1.write(newdata)
            elif len(data)==10:
                start,mid,mid1,end=data[0],data[1],data[2],data[3]
                midx,midy,mid1x,mid1y,endx,endy=data[4],data[5],data[6],data[7],data[8],data[9]
                midhandxyz=locate_hand_in_3d(1-float(midx),float(midy),mid,videopath)
                mid1handxyz=locate_hand_in_3d(1-float(mid1x),float(mid1y),mid1,videopath)
                endhandxyz=locate_hand_in_3d(1-float(endx),float(endy),end,videopath)
                
                midhandx,midhandy,midhandz=transform_hand_end_to_start(int(start),int(mid),midhandxyz[0],midhandxyz[1],midhandxyz[2],videopath)
                mid1handx,mid1handy,mid1handz=transform_hand_end_to_start(int(mid),int(mid1),mid1handxyz[0],mid1handxyz[1],mid1handxyz[2],videopath)

                endhandx,endhandy,endhandz=transform_hand_end_to_start(int(mid1),int(end),endhandxyz[0],endhandxyz[1],endhandxyz[2],videopath)
                
                newdata=data[0]+','+data[1]+','+data[2]+','+data[3]\
                    +','+str(midhandx)+','+str(midhandy)+','+str(midhandz)\
                        +','+str(mid1handx)+','+str(mid1handy)+','+str(mid1handz)\
                         +','+str(endhandx)+','+str(endhandy)+','+str(endhandz)+'\n'
                        
                f1.write(newdata)
                
            else:
                print('error')
                
        f1.close()
        f.close()

            
    
    
    

