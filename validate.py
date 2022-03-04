import argparse
import os
import sys
import numpy as np 
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import cv2
from data_utils.validateDataLoader import validateDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint
from model.baseline import Baseline as Base
import provider
import open3d as o3d

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    parser.add_argument('--data_path', type=str, default='', help='Benchmark path')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
    parser.add_argument('--num_workers', type=int, default=0, help='Worker Number [default: 0]')
    parser.add_argument('--model_name', default='', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    parser.add_argument('--length', action='store_true', default=1, help='Whether to use normal information [default: False]')
    parser.add_argument('--vis', action='store_true', default=0, help='Whether to use normal information [default: False]')

    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result_path='./result/'
    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_Baseline-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint, checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = args.data_path

    VALID_DATASET = validateDataLoader(root=DATA_PATH,num=args.num_point)
    finalvalidDataLoader = torch.utils.data.DataLoader(VALID_DATASET, batch_size=args.batchsize,shuffle=False)
    logger.info("The number of test data is: %d", len(VALID_DATASET))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''

    classifier = Base(args.batchsize).to(device).eval()
    
  
    print('Load CheckPoint...')
    logger.info('Load CheckPoint')
    #need to cancel comment!!!!!!!!!!!!!!!!!!!!
    checkpoint = torch.load(args.checkpoint)
    classifier.load_state_dict(checkpoint['model_state_dict'])


    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    
    for batch_id, data in tqdm(enumerate(finalvalidDataLoader, 0), total=len(finalvalidDataLoader), smoothing=0.9):
           
        
            gt_xyz,pointcloud,geometry,LENGTH,clipsource= data
            
            pointcloud=pointcloud.transpose(3,2)
            
            gt_xyz,pointcloud,geometry=gt_xyz.to(device),pointcloud.to(device),geometry.to(device)
            
            tic=cv2.getTickCount()
            pred = classifier(pointcloud[:,:,:3,:],pointcloud[:,:,3:,:],geometry,LENGTH.max().repeat(torch.cuda.device_count()).to(device))

            toc=cv2.getTickCount()-tic    
            toc /= cv2.getTickFrequency()
            print('speed:',LENGTH/toc,'FPS')
            model_path = os.path.join('./results', args.model_name, clipsource[0][0],clipsource[1][0])
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path=os.path.join(model_path,clipsource[2][0]+'-'+clipsource[3][0]+'.txt')
            gt_path=os.path.join(model_path,clipsource[2][0]+'-'+clipsource[3][0]+'_gt.txt')
            np.savetxt(gt_path,gt_xyz[0][:len(pred)].cpu().numpy())

            with open(result_path, 'w') as f:
                for xx in pred:
                    
                    def dcon(x):
                        resultlist=torch.linspace(-1,1,1024*5).cuda()
                        x=x/x.max()

                        x[torch.where(x<=0.5)]=0

                        return (x*resultlist).sum()/x.sum()
  

                    data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                    f.write(data+'\n')
          
                
            
            
    logger.info('End of evaluation...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
