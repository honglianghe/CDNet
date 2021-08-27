
import utils
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from skimage import measure
from loss import LossVariance, MulticlassDiceLoss, BoundaryLoss, FocalLoss2d, RobustFocalLoss2d
from skimage import morphology, io, color, measure
import cv2
import os

import warnings
warnings.filterwarnings("ignore")







def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

global criterion_var
criterion_var = LossVariance()

global criterion_dice
criterion_dice = MulticlassDiceLoss()

global criterion_BoundaryLoss
criterion_BoundaryLoss = BoundaryLoss()

global criterion_FocalLoss2d
criterion_FocalLoss2d = FocalLoss2d()


global criterion_RobustFocalLoss2d
criterion_RobustFocalLoss2d = RobustFocalLoss2d()




import hhl_utils.pytorch_ssim
criterion_ssimLoss = hhl_utils.pytorch_ssim.SSIM(window_size=11,size_average=True)


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def train(train_loader, model, optimizer, criterion, epoch, opt, logger,
          get_process_worktime = 1, get_process_detail = 1,
          accuracy_tensor = 0):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(8)

    results_loss = utils.AverageMeter(opt.model['out_c'] * 2)  # 3


    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        input, weight_map, target0 = sample
        # 3-class: boundary=2；2-class boundary=1
        if (opt.model['multi_class'] == True):
            boundary = 2
        else:
            boundary = 1

        if (target0.shape[1] == 3):
            target1 = target0.detach().cpu().numpy()
            target = np.zeros((target0.shape[0], target0.shape[-2], target0.shape[-1]), dtype=np.uint8)
            color_number = target1.max()
            for j in range(target0.shape[0]):
                target[j, :, :][target1[j, 0, :, :] == color_number] = 0
                target[j, :, :][target1[j, 1, :, :] == color_number] = 1
                target[j, :, :][target1[j, 2, :, :] == color_number] = boundary
            target = torch.from_numpy(target).long()
        else:
            target = target0
            target1 = target0.detach().cpu().numpy()
            num_classes = 3 if opt.model['multi_class'] == True else 1#2
            target_temp = np.zeros((target0.shape[0], num_classes, target0.shape[-2], target0.shape[-1]),
                                   dtype=np.uint8)
            color_number = np.unique(target1)
            for j in range(target0.shape[0]):
                target_temp[j, 0, :, :][target1[j, 0, :, :] == color_number[0]] = 1
                try:
                    target_temp[j, 1, :, :][target1[j, 0, :, :] == color_number[1]] = 1
                    if (num_classes == 3):
                        target_temp[j, 2, :, :][target1[j, 0, :, :] == color_number[2]] = 1
                    else:
                        target_temp[j, 0, :, :][target1[j, 0, :, :] != color_number[0]] = 1
                except:
                    if (num_classes == 3):
                        print('train IndexError: index 1 is out of bounds for axis 0 with size 1')
                
            target0 = torch.from_numpy(target_temp).long()

        
        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()

        if torch.max(target) == 255:
            target = target // np.int(255 / 2)
        if target.dim() == 4:
            target = target.squeeze(1)

        input_var = input.cuda()
        target_var = target.cuda()
        # 3-channel label
        target0_var = target0.float().cuda()

        start_time = time.time()
        # output
        output = model(input_var)

        log_prob_maps = F.log_softmax(output, dim=1)

        loss_map = criterion(log_prob_maps, target_var.long())
        loss_map_ori = loss_map
        

        if opt.model['add_weightMap'] == True:  
            loss_map = loss_map * weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] == 1:
            prob_maps = F.softmax(output, dim=1)
            # label instances in target
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var_labeled = criterion_var(prob_maps, target_labeled.cuda())
            loss_var = loss_var_labeled# + loss_var_2c
            loss = loss_CE + opt.train['alpha'] * loss_var

        elif opt.train['alpha'] == 2:  # just use loss_var
            prob_maps = F.softmax(output, dim=1)
            # label instances in target
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = opt.train['alpha'] * loss_var
        elif opt.train['alpha'] == 3:  # add ssim_loss
            loss_var = torch.ones(1) * -1
            loss = loss_CE
            model_ssimloss = 1
        
        else:  # without add loss_var
            loss_var = torch.ones(1) * -1
            loss = loss_CE
            model_ssimloss = 0
            ssimloss = loss_var


        # =========================================== boundary loss ===========================================
        beta = 1
        if (opt.model['boundary_loss'] == 1):
            boundary_loss = criterion_BoundaryLoss(output, target0_var)
            loss = loss + beta * boundary_loss
        elif(opt.model['boundary_loss'] == 2): #add focal loss
            boundary_loss = criterion_FocalLoss2d(output, target0_var)
            loss = loss + beta * boundary_loss
        elif(opt.model['boundary_loss'] == 3): #add robust focal loss
            boundary_loss = criterion_RobustFocalLoss2d(output, target0_var)
            loss = loss + beta * boundary_loss
        else:  # without add boundary loss
            loss = loss

        # add dice loss
        if opt.model['dice'] == 1:
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss + loss_dice
        elif opt.model['dice'] == 2:  # 只使用dice loss
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss_dice
        
        # add ssimloss
        model_ssimloss = 0
        if model_ssimloss ==1:
            ssimloss = 0
            channel_n = target0_var.shape[1]
            '''
            for c in range(channel_n):
                ssimloss_item = criterion_ssimLoss(prob_maps[:, c, :, :].unsqueeze(1), target0_var[:, c, :, :].unsqueeze(1))
                ssimloss = ssimloss + ssimloss_item
            ssimloss = ssimloss / channel_n
            '''
            #ssimloss = 20*criterion_ssimLoss(prob_maps[:, 2, :, :].unsqueeze(1), target0_var[:, 2, :, :].unsqueeze(1))
            ssimloss = criterion_ssimLoss(prob_maps, target0_var)
            loss = loss + ssimloss

        # measure accuracy and record loss
        # accuracy_tensor = 0
        if (accuracy_tensor == 0):
            # print('array ------------------------------------------------')
            pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
            metrics = utils.accuracy_pixel_level(pred, target.numpy())
        else:
            ##print('tensor------------------------------------------------')
            pred_tensor = torch.argmax(log_prob_maps, dim=1, keepdim=False)
            metrics = utils.accuracy_pixel_level_tensor(pred_tensor, target)
        
        pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1 = metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]
        
        if(loss.type() != 'torch.cuda.FloatTensor'):
            print('loss = {}, loss_CE = {}, loss_var = {}'.format(loss, loss_CE, loss_var))
            loss = loss * torch.ones(1)
            loss_CE = loss_CE * torch.ones(1)
            loss_var = loss_var * torch.ones(1)
        result = [loss.item(), loss_CE.item(), ssimloss.item(), pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1]
        #
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        loss.cuda().backward()

        optimizer.step()

        del input_var, output, target_var, log_prob_maps, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\t Loss {r[0]:.4f}'
                        '\tLoss_CE {r[1]:.4f}'
                        '\tLoss_var {r[2]:.4f}'
                        '\tPixel_Accu {r[3]:.4f}'
                        '\n\t\t\t\t\t\t\t pixel_IoU {r[4]:.4f}'
                        '\tpixel_Recall {r[5]:.4f}'
                        '\tpixel_Precision {r[6]:.4f}'
                        '\tpixel_F1 {r[7]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: \t Loss {r[0]:.4f}'
                '\tLoss_CE {r[1]:.4f}'
                '\tLoss_var {r[2]:.4f}'
                '\tPixel_Accu {r[3]:.4f}'
                '\n\t\t\t\t\t\t\t pixel_IoU {r[4]:.4f}'
                '\tpixel_Recall {r[5]:.4f}'
                '\tpixel_Precision {r[6]:.4f}'
                '\tpixel_F1 {r[7]:.4f}'.format(epoch, opt.train['num_epochs'], r=results.avg))
    
    
    return results.avg












































def validate(val_loader, model, criterion, epoch,
             opt, logger,
             labeled_df_list,
             get_process_worktime = 1, get_process_detail = 1,
             all_img_test = 1, accuracy_tensor = 0):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(6)
    

    results_loss = utils.AverageMeter(opt.model['out_c'] * 2)  # 3

    # switch to evaluate mode
    model.eval()
    start_time = time.time()
    for i, sample in enumerate(val_loader):
        input, weight_map, target0 = sample

        if (opt.model['multi_class'] == True):
            boundary = 2
        else:
            boundary = 1
        #
        if (target0.shape[1] == 3):
            target1 = target0.detach().cpu().numpy()
            target = np.zeros((target0.shape[0], target0.shape[-2], target0.shape[-1]), dtype=np.uint8)
            color_number = target1.max()
            for j in range(target0.shape[0]):
                target[j, :, :][target1[j, 0, :, :] == color_number] = 0
                target[j, :, :][target1[j, 1, :, :] == color_number] = 1
                target[j, :, :][target1[j, 2, :, :] == color_number] = boundary
            target = torch.from_numpy(target).long()
        else:
            target = target0
            target1 = target0.detach().cpu().numpy()
            num_classes = 3 if opt.model['multi_class'] == True else 1#2
            target_temp = np.zeros((target0.shape[0], num_classes, target0.shape[-2], target0.shape[-1]),
                                   dtype=np.uint8)
            color_number = np.unique(target1)
            for j in range(target0.shape[0]):
                target_temp[j, 0, :, :][target1[j, 0, :, :] == color_number[0]] = 1
                try:
                    target_temp[j, 1, :, :][target1[j, 0, :, :] == color_number[1]] = 1
                    if (num_classes == 3):
                        target_temp[j, 2, :, :][target1[j, 0, :, :] == color_number[2]] = 1
                    else:
                        target_temp[j, 0, :, :][target1[j, 0, :, :] != color_number[0]] = 1
                except:
                    if (num_classes == 3):
                        print('train IndexError: index 1 is out of bounds for axis 0 with size 1')
                
            target0 = torch.from_numpy(target_temp).long()

        weight_map = weight_map.float().div(20)
        if weight_map.dim() == 4:
            weight_map = weight_map.squeeze(1)
        weight_map_var = weight_map.cuda()


        if torch.max(target) == 255:
            target = target // np.int(255 / 2)
        if target.dim() == 4:
            target = target.squeeze(1)

        target_var = target.cuda()
        # 3通道的label
        target0_var = target0.float().cuda()
        size = opt.train['input_size']
        overlap = opt.train['val_overlap']


        if all_img_test == 1:
            with torch.no_grad():
                output = model(input.cuda())
        else:
            output = utils.split_forward(model, input, size, overlap, opt)  # opt.model['out_c']
        #
        if (len(output) == 2):
            output = output[0]

        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var.long())


        #if opt.model['add_weightMap'] == 1:
            #loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] == 1:  
            prob_maps = F.softmax(output, dim=1)
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var_labeled = criterion_var(prob_maps, target_labeled.cuda())
            loss_var = loss_var_labeled# + loss_var_2c
            loss = loss_CE + opt.train['alpha'] * loss_var

        elif opt.train['alpha'] == 2:  # just use loss_var
            prob_maps = F.softmax(output, dim=1)
            target_labeled = torch.zeros(target.size()).long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = opt.train['alpha'] * loss_var
        
        elif opt.train['alpha'] == 3:  # add ssim_loss
            loss = loss_CE
            model_ssimloss = 1
        else:
            loss = loss_CE
            model_ssimloss = 0
            
            
        # add dice loss
        if opt.model['dice'] == 1:
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss + loss_dice
        elif opt.model['dice'] == 2:  # 只使用dice loss
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss_dice
        
        
        # add ssimloss
        model_ssimloss = 0
        if model_ssimloss ==1:
            ssimloss = 0
            channel_n = target0_var.shape[1]
            '''
            for c in range(channel_n):
                ssimloss_item = criterion_ssimLoss(prob_maps[:, c, :, :].unsqueeze(1), target0_var[:, c, :, :].unsqueeze(1))
                ssimloss = ssimloss + ssimloss_item
            ssimloss = ssimloss / channel_n
            '''
            #ssimloss = 20*criterion_ssimLoss(prob_maps[:, 2, :, :].unsqueeze(1), target0_var[:, 2, :, :].unsqueeze(1))
            ssimloss = criterion_ssimLoss(prob_maps, target0_var)
            
            loss = loss + ssimloss
        

        
        
        # accuracy_tensor = 0
        # measure accuracy and record loss
        if (accuracy_tensor == 0):
            pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
            metrics = utils.accuracy_pixel_level(pred, target.numpy())
        else:
            pred_tensor = torch.argmax(log_prob_maps, dim=1, keepdim=False)
            metrics = utils.accuracy_pixel_level_tensor(pred_tensor, target)

        pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1 = metrics[0], metrics[1], metrics[2], metrics[3], metrics[4]  #

        results.update([loss.item(), pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1])

        # =======================================================================================================================
        if (i % len(val_loader) == len(val_loader) - 1 and get_process_worktime == 1 and get_process_detail == 1):

            end_time = time.time()
            work_time = end_time - start_time
            work_time = round(work_time, 4)
            print('\t\t\t === validate() accuracy_pixel_level, the i is [{:d}] === each epoch work time is [{:.4f} s].'.format(i, work_time))
            start_time = end_time
        else:
            start_time = time.time()
        # =======================================================================================================================

        del output, target_var, log_prob_maps, loss

    logger.info('\t=> Val Avg:   \tLoss {r[0]:.4f} \tPixel_Acc {r[1]:.4f}'
                '\tPixel_IoU {r[2]:.4f}'
                '\tpixel_Recall {r[3]:.4f}'
                '\tpixel_Precision {r[4]:.4f}'
                '\tpixel_F1 {r[5]:.4f}'.format(r=results.avg))





