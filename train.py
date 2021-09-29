import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.RGBDDataLoader import RGBDDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint,show_point_cloud
from model.baseline import Baseline as Base
import provider
import numpy as np 
from loss import oriloss

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Predictor baseline train')
    parser.add_argument('--batchsize', type=int, default=8, help='batch size in training')
    parser.add_argument('--epoch',  default=30, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0,1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 1024]')
    parser.add_argument('--num_workers', type=int, default=8, help='Worker Number [default: 16]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='NYU', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    parser.add_argument('--datapath', action='store_true', default='', help='The path of dataset')

    return parser.parse_args()
def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    basepath=os.getcwd()
    '''CREATE DIR'''
    experiment_dir = Path(os.path.join(basepath,'experiment'))
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_ModelNet40-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = args.datapath

    TRAIN_DATASET = RGBDDataLoader(root=DATA_PATH,num=args.num_point)
    
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    logger.info("The number of training data is: %d", len(TRAIN_DATASET))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier = Base(args.batchsize).train()

    if torch.cuda.device_count() > 1:
        classifier = torch.nn.DataParallel(classifier)
    
    classifier.to(device)

    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        start_epoch = torch.load(args.pretrain)['epoch']
        classifier.module.load_state_dict(torch.load(args.pretrain)['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0

    def shufflepoint(workspace_point):
        jittered_data = provider.random_scale_point_cloud(workspace_point[:,:, 0:3], scale_low=2.0/3, scale_high=3/2.0)
        jittered_data = provider.shift_point_cloud(jittered_data, shift_range=0.2)
        workspace_point[:, :, 0:3] = jittered_data
        workspace_point = provider.random_point_dropout_v2(workspace_point)
        provider.shuffle_points(workspace_point)

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        params = list(map(id, classifier.module.gru.parameters()))

        trainable_params = []
        trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                            classifier.module.gru.parameters()),
                          'lr': args.learning_rate}]
        trainable_params += [{'params': filter(lambda x:id(x) not in params,classifier.module.parameters()),
                          'lr': args.learning_rate}]
        optimizer =  torch.optim.Adam(trainable_params,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=args.decay_rate)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)
        print('lr=',optimizer.state_dict()['param_groups'][0]['lr'])
        totalloss = 0
        scheduler.step()
        for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            
            gt_xyz,pointcloud,geometry,LENGTH,name= data
            

            pointcloud=pointcloud.transpose(3,2)
            
            gt_xyz,pointcloud,geometry=gt_xyz.to(device),pointcloud.to(device),geometry.to(device)

            optimizer.zero_grad()
            
            if gt_xyz.size()[0]!=args.batchsize:
                break
            pred = classifier(pointcloud[:,:,:3,:],pointcloud[:,:,3:,:],geometry,LENGTH.max().repeat(torch.cuda.device_count()).to(device))
            
            
            loss =oriloss(pred,gt_xyz,LENGTH,device)


            loss.backward()
            optimizer.step()
            global_step += 1
            totalloss=loss+totalloss
            
            #print('\r Loss: %f' % float(loss))
            nnum=1
            if (batch_id+1)==nnum:
            
                print('\r Loss: %f' % float(totalloss/nnum))
                preloss=totalloss
            elif (batch_id+1)%nnum==0:
                print('\r Loss: %f' % float((totalloss-preloss)/nnum))
                preloss=totalloss
            else:
                None

        if (global_epoch+1)%1==0 :
            save_checkpoint(
                global_epoch + 1,
                classifier.module,
                optimizer,
                str(checkpoints_dir),
                args.model_name)
            print('Saving model....')

        print('\r Loss: %f' % loss.data)
        logger.info('Loss: %.2f', totalloss)
    


        global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
