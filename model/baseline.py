#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch as t
import numpy as np
import torch.nn.functional as F
from torch.nn.modules.activation import Sigmoid
from utils.pointconv_util import PointConvDensitySetAbstraction

class pointconvbackbone(nn.Module):
    def __init__(self):
        super(pointconvbackbone, self).__init__()
        feature_dim = 3
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth = 0.4, group_all=True)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        return x
    
    
class pointconvbackbonetest(nn.Module):
    def __init__(self):
        super(pointconvbackbonetest, self).__init__()
        feature_dim = 3
        self.sa1 = PointConvDensitySetAbstraction(npoint=1, nsample=32, in_channel=feature_dim + 3, mlp=[64, 64, 128], bandwidth = 0.1, group_all=False)
        
    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        l1_xyz, l1_points = self.sa1(xyz, feat)
        x = l1_points.view(B, 128)
        return x
    
    
    
class Baseline(nn.Module):
    def __init__(self,batch):
        super(Baseline, self).__init__()
        num_LSTM=2
        midfc_channel=1024
        self.hind=midfc_channel
            
        self.backbone=pointconvbackbone()
        self.mlp_semantic=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel),
            nn.BatchNorm1d(midfc_channel),
            nn.ReLU(),
            nn.Linear(midfc_channel, midfc_channel),

        )

        self.mlp_geometry=nn.Sequential(
            nn.Linear(18, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )
        self.fine=nn.Sequential(
            nn.Linear(midfc_channel*2, midfc_channel),
                nn.BatchNorm1d(midfc_channel),
                nn.ReLU(),               
                nn.Linear(midfc_channel, midfc_channel),)
        
        self.temporalnet=nn.LSTM(midfc_channel,midfc_channel,num_LSTM)



        self.initiala1=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )
        self.initiala2=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*2),
            nn.BatchNorm1d(midfc_channel*2),
            nn.ReLU(),
            nn.Linear(midfc_channel*2, midfc_channel)
        )



        self.contin1=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*3),
            nn.BatchNorm1d(midfc_channel*3),
            nn.ReLU(),
            nn.Linear(midfc_channel*3, midfc_channel)
        )
        self.contin2=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*3),
            nn.BatchNorm1d(midfc_channel*3),
            nn.ReLU(),
            nn.Linear(midfc_channel*3, midfc_channel)
        )


        

        self.x=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        self.y=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )
        self.z=nn.Sequential(
            nn.Linear(midfc_channel, midfc_channel*5),
            nn.BatchNorm1d(midfc_channel*5),
            nn.ReLU(),
            nn.Linear(midfc_channel*5, 1024*5)
            )

        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight(), gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
             
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight.item(), 1)
                nn.init.constant_(m.bias.item(), 0)  

        

        
    def forward(self,pointxyz,pointfeat,geometry,LEGHTN):
        
        
        final_RES=self.each_feature(pointxyz,pointfeat,geometry,LEGHTN)
         
        
        return final_RES
        
        

    def each_feature(self, pointxyz,pointfeat,geometry,LEGHTN):

        batch_size=geometry.shape[0]
        predictlist=[]

        for sequences in range(int(LEGHTN[0])):
            eachsequences_feature=self.backbone(pointxyz[:,sequences,:,:].float(),pointfeat[:,sequences,:,:].float())
            sematic_feature = self.mlp_semantic(eachsequences_feature)
            geometry_feature = self.mlp_geometry(geometry[:,sequences,:].float())
            feature=self.fine(t.cat((sematic_feature,geometry_feature),-1)).unsqueeze(0)

            if sequences==0:
                cinit=t.cat((self.initiala1(feature.squeeze(0)).unsqueeze(0),self.initiala2(feature.squeeze(0)).unsqueeze(0),\
                    ),0)
                hout=t.cat((self.contin1(feature.squeeze(0)).unsqueeze(0),self.contin2(feature.squeeze(0)).unsqueeze(0),\
                 ),0)


                output,(hout,cout)=self.temporalnet(feature,(hout,cinit))
                
            else:
                
                output,(hout,cout)=self.temporalnet(feature,(hout,cout))

            predictlist.append(t.cat((self.x(output[-1]).unsqueeze(1),self.y(output[-1]).unsqueeze(1),self.z(output[-1]).unsqueeze(1)),1))
        
        return predictlist
