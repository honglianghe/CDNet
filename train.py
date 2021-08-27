import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
import pandas as pd
import random
from skimage import measure
import logging
from tensorboardX import SummaryWriter

import utils
from data_folder import DataFolder
from options import Options
# from my_transforms import get_transforms
from loss import LossVariance, MulticlassDiceLoss, BoundaryLoss
import train_util
import time


import os


path = r'D:\EditSoftware\DemoTest\PHDPaperDemo\CDNet'
os.chdir(path)


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def seed_torch(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    # get_process_worktime:  ; get_process_detail:  ; accuracy_tensor:
    global get_process_worktime, get_process_detail, accuracy_tensor
    get_process_worktime = 1
    get_process_detail = 0
    accuracy_tensor = 0

    start_all_time = time.time()

    global opt, best_iou, best_loss, num_iter, tb_writer, logger, logger_results
    best_iou = 0
    best_loss = 10000
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    if (opt.train['seed'] > 0):
        seed_torch(seed=int(opt.train['seed']))
        print('================================= seed = {} ================================='.format(opt.train['seed']))

    global all_img_test
    all_img_test = opt.all_img_test

    global branch
    # branch = 2
    branch = opt.train['branch']

    do_direction = opt.model['direction']

    # 如果opt.model['direction'] 为1，则执行 my_transforms_direction.py 中的代码
    if (do_direction == 1):
        from my_transforms_direction import get_transforms
        import train_util_dam
    else:
        from my_transforms import get_transforms

    #tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))  # hhl本地无法使用 需要注释掉
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpu'])
    os.environ['CUDA_CACHE_PATH'] = '~/.cudacache'

    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    # ----- define criterion ----- #
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    if (opt.model['multi_class'] == True):
        criterion_CE = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.6, 0.3, 0.1])).float(),
                                                 reduction='none').cuda()
    else:
        criterion_CE = torch.nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([0.8, 0.2])).float(),
                                                 reduction='none').cuda()
    criterion_CE = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    print(criterion_CE)

    gpu_str = ''
    for gpu_i in opt.train['gpu']:
        gpu_str = gpu_str + str(gpu_i)
    # log
    logExl_path = './experiments/' + opt.dataset + '/'
    logExl_name = '{}_logExl_gpu{}.csv'.format(opt.dataset, gpu_str)
    logExl_PathAndName = logExl_path + logExl_name
    # 判断是否存在 MultiOrgan_logExl.csv， 如果否，则生成一个新文件
    # acc, iou, recall, precision, F1
    if (os.path.exists(logExl_PathAndName) == False):
        logExl = pd.DataFrame(
            columns=['Model', 'Epoch', 'Test_epoch', 'input_size', 'val_overlap',  # 0-4 (0开始)
                     'batch_size', 'multi_class', 'add_weightMap', 'alpha', 'dice', 'boundary_loss',  # 5-10 (0开始)
                     'MSEloss', 'direction',  # 11-12 (0开始)
                     'backbone', 'pretrained', 'LossName', 'seed', 'early_stop',  # 13-17 (0开始)
                     'scheduler', 'step', 'optimizer', 'lr', 'lr_decay',  # 18-22 (0开始)
                     'postproc', 'min_area', 'radius', 'validation', 'groundtruth', 'AllImgTest',  # 23-28 (0开始)
                     'random_resize', 'random_color', 'random_affine', 'horizontal_flip',  # 29-32 (0开始)
                     'random_elastic', 'random_rotation', 'random_chooseAug', 'random_crop', 'normalize',  # 33-37 (0开始)
                     't1_pixel_acc', 't1_pixel_IoU', 't1_pixel_recall', 't1_pixel_precision', 't1_pixel_F1',  # 38-60 t1
                     't1_recall', 't1_precision', 't1_F1', 't1_Dice', 't1_IoU',
                     't1_Hausdorff', 't1_AJI', 't1_AJI_sklearn', 't1_AJI_h', 't1_Dice_h',
                     't1_Dice2_h','t1_dq','t1_sq','t1_pq',
                     't1_ana_FP','t1_ana_FN','t1_P_less','t1_P_more', # 
                     't2_pixel_acc', 't2_pixel_IoU', 't2_pixel_recall', 't2_pixel_precision', 't2_pixel_F1',  # 61-83 t2
                     't2_recall', 't2_precision', 't2_F1', 't2_Dice', 't2_IoU',
                     't2_Hausdorff', 't2_AJI', 't2_AJI_sklearn', 't2_AJI_h', 't2_Dice_h',
                     't2_Dice2_h','t2_dq','t2_sq','t2_pq',
                     't2_ana_FP','t2_ana_FN','t2_P_less','t2_P_more', #
                     ])

        logExl.to_csv(logExl_path + logExl_name, index=False)
        logExl_columns = logExl.columns
        logExl_number = logExl.shape[0]
    else:
        print('{:s}_logExl.csv have been exist'.format(opt.dataset))
        logExl = pd.read_csv(logExl_PathAndName)
        logExl_columns = logExl.columns
        logExl_number = logExl.shape[0]
    print('opt.train[trans_train] = ', opt.train['trans_train'])
    trans_train_RRe = 1 if '_isRRe' in opt.train['trans_train'] else 0  # random_resize
    trans_train_RCo = 1 if '_isRCo' in opt.train['trans_train'] else 0  # random_color
    trans_train_RA = 1 if '_isRA' in opt.train['trans_train'] else 0  # random_affine
    trans_train_HF = 1 if '_isHF' in opt.train['trans_train'] else 0  # horizontal_flip
    trans_train_RE = 1 if '_isRE' in opt.train['trans_train'] else 0  # random_elastic
    trans_train_RRo = 1 if '_isRRo' in opt.train['trans_train'] else 0  # random_rotation
    trans_train_CAu = 1 if '_isCAu' in opt.train['trans_train'] else 0  # RandomChooseAug
    trans_train_RCr = 1 if '_isRCr' in opt.train['trans_train'] else 0  # random_crop
    trans_train_Nor = 1 if '_isNorm' in opt.train['trans_train'] else 0  # normalize
    multi_class = 1 if '_3c' in opt.model['exp_filename'] else 0  # multi_class

    log_eachItem = [opt.model['modelName'], opt.train['num_epochs'], 'Test_epoch', opt.train['input_size'],
                    opt.train['val_overlap'],  # 0-4 (0开始)
                    opt.train['batch_size'], multi_class, opt.model['add_weightMap'], opt.train['alpha'],
                    opt.model['dice'], opt.model['boundary_loss'],
                    opt.model['mseloss'], opt.model['direction'],
                    opt.model['backbone'], opt.model['pretrained'], opt.model['LossName'], opt.train['seed'],
                    opt.train['early_stop'],
                    opt.train['scheduler'], opt.train['step'], opt.train['optimizer'], opt.train['lr'],
                    opt.train['lr_decay'],
                    'postproc', 'min_area', 'radius', opt.train['validation'], 'groundtruth', opt.all_img_test,
                    trans_train_RRe, trans_train_RCo, trans_train_RA, trans_train_HF, trans_train_RE, trans_train_RRo,
                    trans_train_CAu, trans_train_RCr, trans_train_Nor,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # 23个0 38-60
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # 23个0 61-83
                    ]
    # 36-48, 49-61
    #print(len(log_eachItem))
    #print(log_eachItem)
    #print(len(logExl.columns))
    logExl.loc[logExl_number] = log_eachItem
    logExl.to_csv(logExl_path + logExl_name, index=False)

    haveModel = 0
    if (haveModel == 1):
        model = utils.chooseModel(opt)
        pthfile = r'D:\EditSoftware\DemoTest\DeepDemo\Nuclei_Seg2021\experiments\MultiOrgan\0_UNet[None][adam]_sche[None]_3c_input208over80bs8_e30_seed2021ms_addWM_br3_Dice1_b0_estop7_isVal_a0\checkpoints\checkpoint_best.pth.tar'
        checkpoint = torch.load(pthfile)
        model.load_state_dict(checkpoint['state_dict'], False)  # False True

    else:
        model = utils.chooseModel(opt)

    model = nn.DataParallel(model)
    model = model.cuda()

    torch.backends.cudnn.benchmark = True  # 如果（每次 iteration）输入数据维度或类型上变化不大,可以增加效率；反之，则会降低效率。

    # ----- define optimizer ----- #
    optimizer, scheduler = utils.get_optimizer(args=opt, model=model)

    if (haveModel == 1):
        optimizer.load_state_dict(checkpoint['optimizer'])

    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()
    
    
    
    
    input_train_numbers = 16
    input_scale = 300
    train_name = 'train' + '_' + str(input_scale) 
    #train_name = 'train' + '_16_' + str(input_scale) 
    
    # train_name处原为'train'
    data_transforms = {train_name: get_transforms(opt.transform['train']),
                       'val': get_transforms(opt.transform['val'])}

    # ----- load data ----- #
    dsets = {}  # GlaS MultiOrgan MultiOrgan_Val MoNuSeg CoNSeP 2018DSB CPM2017 COVID19

    if opt.train['validation'] == 1:
        for x in ['train', 'val']:  # 'val'
            #x = 'train_ori' if x == 'val' else train_name #'train'
            if (opt.dataset == 'CPM2017') and (x == 'val'):
                print('=*=' * 10)
                img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], 'test')
                target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], 'test'+'_ins')  # +'_ins'
                weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], 'test')
            else:
                img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
                x_ins = 'train_ins_ori' if x == 'train_ori' else x +'_ins'
                target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x_ins)  #x +'_ins'
                weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)

            dir_list = [img_dir, weight_map_dir, target_dir]
            print(dir_list)
            if opt.dataset == 'CPM2017' or opt.dataset == 'MultiOrgan':
                #post_fix = ['weight.png', 'label.png']  # mask  label.png seg_colored.png
                post_fix = ['weight.png', 'label.mat'] # [ CPM2017 label.mat],   others: label.npy

            else:
                #post_fix = ['weight.png', 'label.png']  # mask  label.png seg_colored.png
                post_fix = ['weight.png', 'label.npy']
            
            num_channels = [3, 1, 3]   # [3, 1, 1]
            
            if('_ins' in target_dir):
                print('**'*30)
                #x = 'val'+'_ins' if x == 'train_ori' else 'train'+'_ins'
                if x == 'train_ori':
                    x = 'val'#+'_ins'
            else:
                print('=='*30)
                # x = 'val' if x == 'train_ori' else 'train'
                if x == 'train_ori':
                    x = 'val'
            print(x)
            dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])
            print(dir_list)

        train_loader = DataLoader(dsets[train_name], batch_size=opt.train['batch_size'], shuffle=True, pin_memory=True,
                                  num_workers=opt.train['workers'], drop_last=True)

        val_loader = DataLoader(dsets['val'], batch_size=1, shuffle=False, num_workers=opt.train['workers'],
                                drop_last=True)


        labeled_df_list = []

    else:
        x = train_name
        img_dir = '{:s}/{:s}'.format(opt.train['img_dir'], x)
        target_dir = '{:s}/{:s}'.format(opt.train['label_dir'], x)  # +'_ins'
        weight_map_dir = '{:s}/{:s}'.format(opt.train['weight_map_dir'], x)

        dir_list = [img_dir, weight_map_dir, target_dir]

        #if opt.dataset != 'GlaS':
        if opt.dataset == 'CPM2017' or opt.dataset == 'MultiOrgan':
            post_fix = ['weight.png', 'label.png']  # mask  label.png seg_colored.png
            #post_fix = ['weight.png', 'label.mat'] # [ CPM2017 label.mat],   others: label.npy
        
        else:
            post_fix = ['weight.png', 'label.png']  # mask  label.png seg_colored.png
            #post_fix = ['weight.png', 'label.npy']
        if opt.dataset == 'BBBC039V1':
            num_channels = [1, 1, 3] #[3, 1, 1] #
        else:
            num_channels = [3, 1, 3] #[3, 1, 1] #

        print('dir_list = ', dir_list)
        print('post_fix = ', post_fix)
        dsets[x] = DataFolder(dir_list, post_fix, num_channels, data_transforms[x])
        train_loader = DataLoader(dsets[train_name], batch_size=opt.train['batch_size'], shuffle=True, pin_memory=True,
                                  num_workers=opt.train['workers'])
        logger.info('=======================No validation=======================')

    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            best_iou = checkpoint['best_iou']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    if opt.train['early_stop'] > 0:
        print('===================== do early_stop = {} ====================='.format(opt.train['early_stop']))

        # opt.train['validation'] = 1

        early_stopping = utils.EarlyStopping(patience=opt.train['early_stop'], verbose=False, delta=0)

    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch + 1, opt.train['num_epochs']))
        start_time = time.time()
        if (do_direction == 1):
            train_loss, train_loss_mse, train_loss_mse_FP, train_loss_mse_FN, train_loss_ce, train_loss_var, \
            train_pixel_acc, train_iou, train_Recall, train_Precision, train_F1 = train_util_dam.train(train_loader,
                                                                                                       model, optimizer,
                                                                                                       criterion, epoch,
                                                                                                       opt, logger,
                                                                                                       get_process_worktime=get_process_worktime,
                                                                                                       get_process_detail=get_process_detail,
                                                                                                       accuracy_tensor=accuracy_tensor)
        else:
            train_results = train_util.train(train_loader, model, optimizer, criterion, epoch, opt, logger,
                                             get_process_worktime=get_process_worktime,
                                             get_process_detail=get_process_detail,
                                             accuracy_tensor=accuracy_tensor)
            train_loss, train_loss_ce, train_loss_var, train_pixel_acc, train_iou, train_Recall, train_Precision, train_F1 = train_results
            train_loss_mse = 0.0
            train_loss_mse_FP = 0.0
            train_loss_mse_FN = 0.0

        # =======================================================================================================================
        if (epoch % 5 == 0 and get_process_worktime == 1):
            end_time = time.time()
            work_time = end_time - start_time
            work_time = round(work_time, 4)
            print('\t\t\t =======      train()      ========  each epoch work time is [{:.4f} s].'.format(work_time))
            start_time = end_time
        # =======================================================================================================================

        if opt.train['validation'] == 1 and do_direction == 0:

            # evaluate on validation set
            with torch.no_grad():
                val_loss, val_pixel_acc, val_iou, val_Recall, val_Precision, val_F1 = train_util.validate(val_loader,
                                                                                                          model,
                                                                                                          criterion,
                                                                                                          epoch,
                                                                                                          opt, logger,
                                                                                                          labeled_df_list,
                                                                                                          get_process_worktime=get_process_worktime,
                                                                                                          get_process_detail=get_process_detail,
                                                                                                          all_img_test=all_img_test,
                                                                                                          accuracy_tensor=accuracy_tensor)
                # train_util.validate(val_loader, model, model_branch, criterion)
        elif opt.train['validation'] == 1 and do_direction == 1:

            # evaluate on validation set
            with torch.no_grad():
                # hhl 用于在n个epoch之后执行obj的metrics计算
                # #do_object_metric = 1 if epoch>int(opt.train['num_epochs']*0.75) else 0  # time consuming
                do_object_metric = 0
                val_loss, val_loss_mse, val_loss_mse_FP, val_loss_mse_FN, val_pixel_acc, val_iou, val_Recall, \
                val_Precision, val_F1, val_obj_recall, val_obj_precision, val_obj_F1, val_obj_dice, val_obj_iou, \
                val_obj_haus, val_obj_AJI = train_util_dam.validate(val_loader, model, criterion, opt, logger,
                                                                    get_process_worktime=get_process_worktime,
                                                                    get_process_detail=get_process_detail,
                                                                    all_img_test=all_img_test,
                                                                    accuracy_tensor=accuracy_tensor,
                                                                    do_object_metric=do_object_metric)

        else:
            val_loss = train_loss
            val_pixel_acc = train_pixel_acc
            val_iou = train_iou
            val_Recall = train_Recall
            val_Precision = train_Precision
            val_F1 = train_F1

            val_obj_iou = val_iou

        # check if it is the best accuracy
        is_best = val_iou > best_iou
        # is_best = val_loss < best_loss
        best_iou = max(val_iou, best_iou)
        best_loss = min(val_loss, best_loss)

        # =======================================================================================================================
        if (epoch % 5 == 0 and get_process_worktime == 1):
            end_time = time.time()
            work_time = end_time - start_time
            work_time = round(work_time, 4)
            print('\t\t\t =======    validate()     ======== each epoch work time is [{:.4f} s].'.format(work_time))
            start_time = end_time
        # =======================================================================================================================

        if scheduler is None:
            prev_lr = utils.adjust_learning_rate(args=opt, optimizer=optimizer, epoch=epoch)  # (无效的，没有进行迭代)
        else:
            prev_lr = optimizer.param_groups[0]['lr']
            if 'ReduceLROnPlateau' == opt.train['scheduler']:
                scheduler.step(val_loss)
            else:
                scheduler.step()
            print(
                f"===================== Updating learning rate from {prev_lr} to {optimizer.param_groups[0]['lr']} =====================")

        
        if (opt.train['num_epochs'] - (epoch) <= 10 and epoch > 40):
            cp_flag = 1
            print('epoch = {}, cp_flag = {}'.format(epoch, cp_flag))
        else:
            cp_flag = 0
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_iou': best_iou,
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, epoch, is_best, opt.train['save_dir'], 'Main', cp_flag)

        # save the training results to txt files
        logger_results.info(
            '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
                .format(epoch + 1, train_loss, train_loss_mse, train_loss_ce, train_loss_var, train_pixel_acc,
                        train_iou, train_Recall, train_Precision, train_F1, val_loss, val_pixel_acc, val_iou,
                        val_Recall, val_Precision, val_F1))

        # 本地运行时需要进行注释 hhl20200721
        # tensorboard logs
        #tb_writer.add_scalars('epoch_losses', {'train_loss': train_loss, 'train_loss_ce': train_loss_ce, 'train_loss_var': train_loss_var, 'val_loss': val_loss}, epoch)
        #tb_writer.add_scalars('epoch_accuracies', {'train_pixel_acc': train_pixel_acc, 'train_iou': train_iou, 'val_pixel_acc': val_pixel_acc, 'val_iou': val_iou}, epoch)

        if opt.train['early_stop'] > 0:
            # early_stopping(val_loss, epoch)
            early_stopping(-val_F1 - val_iou, epoch)
            if early_stopping.early_stop == True:
                print("epoch = {} Early stopping...".format(epoch))
                break

    # 本地需要进行注释 hhl
    #tb_writer.close()


    end_all_time = time.time()
    work_all_time = end_all_time - start_all_time
    work_all_time = round(work_all_time, 3)
    work_minute_time = work_all_time / 60.0
    logger.info('This model is: [{}].\t add_weightMap = {}.\t The time spent is [{:.3f} min].'
                .format(opt.model['modelName'], opt.model['add_weightMap'], work_minute_time))


def save_checkpoint(state, epoch, is_best, save_dir, branch_value, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        if branch_value == 'Main':
            branch_value = ''
            shutil.copyfile(filename, '{:s}/checkpoint{:s}_{:d}.pth.tar'.format(cp_dir, branch_value, epoch + 1))
        else:
            shutil.copyfile(filename, '{:s}/checkpoint{:s}_{:d}.pth.tar'.format(cp_dir, branch_value, epoch + 1))

    if is_best:
        if branch_value == 'Main':
            branch_value = ''
            shutil.copyfile(filename, '{:s}/checkpoint{:s}_best.pth.tar'.format(cp_dir, branch_value))
        else:
            shutil.copyfile(filename, '{:s}/checkpoint{:s}_best.pth.tar'.format(cp_dir, branch_value))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        # epoch + 1, train_loss, train_loss_ce, train_loss_var,
        # train_pixel_acc, train_iou, train_Recall, train_Precision, train_F1, val_loss, val_pixel_acc, val_iou, val_Recall, val_Precision, val_F1
        logger_results.info(
            'epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_pixel_acc\ttrain_iou\ttrain_Recall\ttrain_Precision\ttrain_F1\t'
            'val_loss\tval_pixel_acc\tval_iou\tval_Recall\tval_Precision\tval_F1')

    return logger, logger_results


if __name__ == '__main__':
    main()
