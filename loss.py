import argparse
import os
import sys
import numpy as np 
import torch as t


def generatepred(x):
        resultlist=t.linspace(-1,1,1024*5).cuda()
        x=x/x.max(1)[0].unsqueeze(-1)
        for i in range(3):
                x[i][t.where(x[i]<0.5)]=0
        return (x*resultlist).sum(1)/x.sum(1)

def calculate(x,y):

        pred=generatepred(x)
        loss=((pred-y)**2).sum()

        return loss


def oriloss(pred,gt,length,device):
    batch=gt.size()[0]
    loss=[]
    for i in range(batch):
        
        for pred_xyz in range(length[i]):
            loss.append((calculate(pred[pred_xyz][i],gt[i][pred_xyz])*(2-pred_xyz/length[i])))
    return sum(loss)/batch

