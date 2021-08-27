import utils
import time
import numpy as np
import pandas as pd
from skimage import io
from scipy import ndimage as ndi  # do_object_metric hhladd
import skimage.morphology as morph  # do_object_metric hhladd
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from skimage import measure
from loss import LossVariance, MulticlassDiceLoss, BoundaryLoss, FocalLoss2d, RobustFocalLoss2d, WeightMulticlassDiceLoss
import copy
import os

global criterion_var
criterion_var = LossVariance()

global criterion_dice
criterion_dice = MulticlassDiceLoss()

global criterion_weightdice
criterion_weightdice = WeightMulticlassDiceLoss()

global criterion_BoundaryLoss
criterion_BoundaryLoss = BoundaryLoss()

global criterion_FocalLoss2d
criterion_FocalLoss2d = FocalLoss2d()

global criterion_RobustFocalLoss2d
criterion_RobustFocalLoss2d = RobustFocalLoss2d()

global criterion_MSE
criterion_MSE = torch.nn.MSELoss()


def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def train(train_loader, model, optimizer, criterion, epoch, opt, logger,
          get_process_worktime=1, get_process_detail=1,
          accuracy_tensor=0):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(11)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        # False means label is angle，True means label is direction class
        direction_label = True  #  True  False
        

        if (opt.model['direction'] == 1 and opt.model['mseloss'] == 1):
            input, weight_map, target0, target_point0, target_direction0 = sample
        elif(opt.model['direction'] == 1 and opt.model['mseloss'] == 0):
            input, weight_map, target0, target_direction0 = sample
        elif(opt.model['direction'] == 0 and opt.model['mseloss'] == 1):
            input, weight_map, target0, target_point0 = sample
        else:
            input, weight_map, target0 = sample

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
            target = torch.from_numpy(target).float()  # long()
        else:
            target = target0
            target1 = target0.detach().cpu().numpy()
            num_classes = 3 if opt.model['multi_class'] == True else 1  # 2
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
                    if (num_classes != 1):
                        print('train IndexError: index 1 is out of bounds for axis 0 with size 1')
            target0 = torch.from_numpy(target_temp).float()  # long()

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
        # 3-channel label   0:background, 1:inside; 2:boundary
        target0_var = target0.float().cuda()
        
        # target_point0 ==================================================================================
        if(opt.model['mseloss'] == 1):
            target_point = target_point0.float()  # long()


        # target_direction0 ==================================================================================
        if (opt.model['direction'] == 1 and target_direction0.dim() == 3 and direction_label == True):
            target_direction = copy.deepcopy(target_direction0)
            direction_classes = opt.direction_classes
            target_direction_temp = torch.zeros((target_direction0.shape[0], direction_classes,
                                                 target_direction0.shape[-2], target_direction0.shape[-1]),
                                                dtype=torch.float)

            for j in range(target_direction0.shape[0]):
                unique_list = torch.unique(target_direction0[j]).detach().cpu().numpy()
                unique_number = len(unique_list)

                if (unique_number > 1):  # Prevents images from appearing without instances

                    for k in unique_list:

                        target_direction_temp[j, k, :, :][target_direction0[j, :, :] == k] = 1
                        target_direction_temp[j, k, :, :][((target[0, :, :] == 1) + (target[0, :, :] == 2)) == 0] = 0
                else:
                    target_direction_temp[j, 0, :, :][target_direction0[j, :, :] == unique_list[0]] = 1
            target_direction0 = target_direction_temp



        start_time = time.time()

        output_all = model(input_var)


        # print('output_all.len = {}'.format(len(output_all)))
        if (len(output_all) == 3):
            # print('len(output) = {}'.format(len(output)))
            output = output_all[0]
            output_point = output_all[1]
            output_direction = output_all[2]
        elif (len(output_all) == 2):
            output = output_all[0]
            #output_point = output_all[1]
            if (opt.model['mseloss'] == 1):
                output_point = output_all[1]
            elif (opt.model['direction'] == 1):
                output_direction = output_all[1]
        elif (len(output_all) == 1):
            output = output_all

        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var.long())  # long()

        if opt.model['add_weightMap'] == True:
            loss_map *= weight_map_var
        loss_CE = loss_map.mean()

        if opt.train['alpha'] == 1:
            prob_maps = F.softmax(output, dim=1)
            target_labeled = torch.zeros(target.size()).float()  # long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = loss_CE + opt.train['alpha'] * loss_var

        elif opt.train['alpha'] == 2:  # just use loss_var
            prob_maps = F.softmax(output, dim=1)
            # label instances in target
            target_labeled = torch.zeros(target.size()).float()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = opt.train['alpha'] * loss_var

        else:  # without add loss_var
            loss_var = torch.ones(1) * -1
            loss = loss_CE

        # =========================================== boundary loss ===========================================
        beta = 1
        if (opt.model['boundary_loss'] == 1):
            boundary_loss = criterion_BoundaryLoss(output, target0_var)
            loss = loss + beta * boundary_loss
        elif (opt.model['boundary_loss'] == 2):  # add focal loss
            boundary_loss = criterion_FocalLoss2d(output, target0_var)
            loss = loss + beta * boundary_loss
        elif (opt.model['boundary_loss'] == 3):  # add robust focal loss
            boundary_loss = criterion_RobustFocalLoss2d(output, target0_var)
            loss = loss + beta * boundary_loss

        else:  # without add boundary loss
            boundary_loss = torch.tensor(0)

        # add dice loss
        if opt.model['dice'] == 1:
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss + loss_dice

        elif opt.model['dice'] == 2:  # just use dice loss
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss_dice
        else:
            loss_dice = torch.tensor(0)



        if (opt.model['direction'] == 1 and direction_label == True):
            output_direction_0 = output_direction
            log_prob_maps_direction = F.log_softmax(output_direction_0, dim=1)
            loss_direction_map = criterion(log_prob_maps_direction, target_direction.cuda().long())  # long()
            if opt.model['add_weightMap'] == True:
                loss_direction_map *= weight_map_var
            loss_direction_CE = loss_direction_map.mean()

            loss = loss + loss_direction_CE
            # print('loss_direction_CE = {}'.format(loss_direction_CE))

            if opt.model['dice'] == 1:
                prob_maps_direction = F.softmax(output_direction_0, dim=1)
                
                use_weightdice = 1
                # criterion_dice     criterion_weightdice
                if opt.model['add_weightMap'] == True and use_weightdice == 1:
                    loss_direction_dice = criterion_weightdice(prob_maps_direction, target_direction0.cuda(), weight_map_var)
                else:
                    loss_direction_dice = criterion_dice(prob_maps_direction, target_direction0.cuda())

                loss = loss + loss_direction_dice


        elif (opt.model['direction'] == 1 and direction_label == False):

            if (len(target_direction0.shape) == 3):
                target_direction0 = target_direction0.unsqueeze(1)
            loss_angle_mse = criterion_MSE(output_direction.cuda(), (target_direction0).cuda().float())
            loss_angle_mse = loss_angle_mse/1000

            
            loss = loss + loss_angle_mse
            loss_direction_CE = loss_angle_mse
            loss_direction_dice = loss_angle_mse



        else:
            loss_direction_CE = torch.tensor(0.0)
            loss_direction_dice = torch.tensor(0.0)
            loss_var = torch.ones(1) * -2

        if (opt.model['mseloss'] == 1):
            if (len(target_point.shape) == 3):
                target_point = target_point.unsqueeze(1)
            loss_mse = criterion_MSE(output_point.cuda(), (target_point).cuda())  # *10    #/255
            loss = loss + loss_mse
        else:
            loss_mse = torch.tensor(0)

        loss = loss.float()

        if(accuracy_tensor == 0 and opt.model['direction'] == 1):
            pred = np.argmax(log_prob_maps_direction.data.cpu().numpy(), axis=1)
            metrics = utils.accuracy_pixel_level(pred, target_direction.numpy())
        
        elif(accuracy_tensor == 1):
            ##print('tensor------------------------------------------------')
            pred_tensor = torch.argmax(log_prob_maps, dim=1, keepdim=False)
            metrics = utils.accuracy_pixel_level_tensor(pred_tensor, target)
        else: # (accuracy_tensor == 0):

            pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
            metrics = utils.accuracy_pixel_level(pred, target.numpy())
        
        
        
        # print(metrics)
        pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1 = metrics[0], metrics[1], metrics[2], metrics[3], \
                                                                         metrics[4]

        result = [loss.item(), loss_direction_CE.item(), loss_direction_dice.item(), loss_mse.item(),
                  loss_CE.item(), loss_var.item(), pixel_accu, pixel_iou, pixel_recall,
                  pixel_precision, pixel_F1]
        results.update(result, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        # print('loss = {}'.format(loss))
        

        loss.cuda().backward()
        optimizer.step()


        del input_var, output, target_var, log_prob_maps, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\t Loss {r[0]:.4f}'
                        '\tloss_direction_CE {r[1]:.4f}'
                        '\tloss_direction_dice {r[2]:.4f}'
                        '\tloss_mse {r[3]:.4f}'
                        '\tLoss_CE {r[4]:.4f}'
                        '\tLoss_var {r[5]:.4f}'
                        '\tPixel_Accu {r[6]:.4f}'
                        '\n\t\t\t\t\t\t\t\t\t pixel_IoU {r[7]:.4f}'
                        '\tpixel_Recall {r[8]:.4f}'
                        '\tpixel_Precision {r[9]:.4f}'
                        '\tpixel_F1 {r[10]:.4f}'.format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: \t Loss {r[0]:.4f}'
                '\tloss_direction_CE {r[1]:.4f}'
                '\tloss_direction_dice {r[2]:.4f}'
                '\tloss_mse {r[3]:.4f}'
                '\tLoss_CE {r[4]:.4f}'
                '\tLoss_var {r[5]:.4f}'
                '\tPixel_Accu {r[6]:.4f}'
                '\n\t\t\t\t\t\t\t\t\t pixel_IoU {r[7]:.4f}'
                '\tpixel_Recall {r[8]:.4f}'
                '\tpixel_Precision {r[9]:.4f}'
                '\tpixel_F1 {r[10]:.4f}'.format(r=results.avg))

    return results.avg



























def validate(val_loader, model, criterion, opt, logger,
             get_process_worktime=1, get_process_detail=1,
             all_img_test=1, accuracy_tensor=0, do_object_metric=0):
    # list to store the losses and accuracies: [loss, pixel_acc, iou ]
    results = utils.AverageMeter(16)  # 9+7

    # switch to evaluate mode
    model.eval()
    start_time = time.time()
    for i, sample in enumerate(val_loader):

        if (opt.model['direction'] == 1 and opt.model['mseloss'] == 1):
            input, weight_map, target0, target_point0, target_direction0 = sample
        elif(opt.model['direction'] == 1 and opt.model['mseloss'] == 0):
            input, weight_map, target0, target_direction0 = sample
        elif(opt.model['direction'] == 0 and opt.model['mseloss'] == 1):
            input, weight_map, target0, target_point0 = sample
        else:
            input, weight_map, target0 = sample


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
            target = torch.from_numpy(target).float()  # long()
        else:
            target = target0
            target1 = target0.detach().cpu().numpy()

            num_classes = 3 if opt.model['multi_class'] == True else 1  # 2
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
                        # !!!morph.erosion(target_temp[j, 1, :, :], morph.disk(opt.post['radius']))
                except:
                    if (num_classes == 3):
                        print('train IndexError: index 1 is out of bounds for axis 0 with size 1')

            target0 = torch.from_numpy(target_temp).float()  # long()

        

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
        # 3通道的label
        target0_var = target0.float().cuda()

        # target_point0 ==================================================================================
        if(opt.model['mseloss'] == 1):
            target_point = target_point0.float()  # long()

        # target_direction0 ==================================================================================
        if (opt.model['direction'] == 1 and target_direction0.dim() == 3):
            target_direction = copy.deepcopy(target_direction0)
            direction_classes = opt.direction_classes
            target_direction_temp = torch.zeros((target_direction0.shape[0], direction_classes,
                                                 target_direction0.shape[-2], target_direction0.shape[-1]),
                                                dtype=torch.float)
            unique_number = torch.unique(target_direction0)

            for j in range(target_direction0.shape[0]):
                for k in range(direction_classes):
                    target_direction_temp[j, k, :, :][target_direction0[j, :, :] == unique_number[k]] = 1
                    target_direction_temp[j, k, :, :][((target[0, :, :] == 1) + (target[0, :, :] == 2)) == 0] = 0

            target_direction0 = target_direction_temp

        size = opt.train['input_size']
        overlap = opt.train['val_overlap']

        # output = utils.split_forward(model, input, size, overlap, opt.model['out_c'])

        if all_img_test == 1:
            with torch.no_grad():
                output_all = model(input_var)
                # output, output_direction, output_point = model(input_var)
        else:
            output_all = utils.split_forward_dam(model, input, size, overlap, opt)

        if (len(output_all) == 3):
            # print('len(output) = {}'.format(len(output)))
            output = output_all[0]
            output_point = output_all[1]
            output_direction = output_all[2]
        elif (len(output_all) == 2):
            output = output_all[0]
            #output_point = output_all[1]
            if (opt.model['mseloss'] == 1):
                output_point = output_all[1]
            elif (opt.model['direction'] == 1):
                output_direction = output_all[1]
        elif (len(output_all) == 1):
            output = output_all

        log_prob_maps = F.log_softmax(output, dim=1)
        loss_map = criterion(log_prob_maps, target_var.long())  # long()


        #if opt.model['add_weightMap'] == 1:
            #loss_map *= weight_map_var
        loss_CE = loss_map.mean()


        if opt.train['alpha'] == 1:
            prob_maps = F.softmax(output, dim=1)
            target_labeled = torch.zeros(target.size()).float()  # long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
                # utils.show_figures((target[k].numpy(), target[k].numpy()==1, target_labeled[k].numpy()))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = loss_CE + opt.train['alpha'] * loss_var

        elif opt.train['alpha'] == 2:  # just use loss_var
            prob_maps = F.softmax(output, dim=1)
            target_labeled = torch.zeros(target.size()).float()  # long()
            for k in range(target.size(0)):
                target_labeled[k] = torch.from_numpy(measure.label(target[k].numpy() == 1))
            loss_var = criterion_var(prob_maps, target_labeled.cuda())
            loss = opt.train['alpha'] * loss_var
        else:
            loss = loss_CE

        # =========================================== boundary loss ===========================================
        beta = 1
        if (opt.model['boundary_loss'] == 1):
            boundary_loss = criterion_BoundaryLoss(output, target0_var)
            loss = loss + beta * boundary_loss
        elif (opt.model['boundary_loss'] == 2):  # add focal loss
            boundary_loss = criterion_FocalLoss2d(output, target0_var)
            loss = loss + beta * boundary_loss
        elif (opt.model['boundary_loss'] == 3):  # add robust focal loss
            boundary_loss = criterion_RobustFocalLoss2d(output, target0_var)
            loss = loss + beta * boundary_loss
        else:
            boundary_loss = torch.tensor(0)

        if opt.model['dice'] == 1:
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss + loss_dice
        elif opt.model['dice'] == 2:  # just use dice loss
            prob_maps = F.softmax(output, dim=1)
            loss_dice = criterion_dice(prob_maps, target0_var)
            loss = loss_dice
        else:
            loss_dice = torch.tensor(0)

        if (opt.model['direction'] == 1):
            output_direction_0 = output_direction
            log_prob_maps_direction = F.log_softmax(output_direction_0, dim=1)
            loss_direction_map = criterion(log_prob_maps_direction, target_direction.cuda().long())  # long()

            if opt.model['add_weightMap'] == True:
                loss_direction_map *= weight_map_var
            loss_direction_CE = loss_direction_map.mean()

            loss = loss + loss_direction_CE
            # print('loss_direction_CE = {}'.format(loss_direction_CE))

            if opt.model['dice'] == 1:
                prob_maps_direction = F.softmax(output_direction_0, dim=1)
                prob_maps_direction[:, 0, :, :] = prob_maps_direction[:, 0, :, :] * prob_maps[:, 0, :, :]
                loss_direction_dice = criterion_dice(prob_maps_direction, target_direction0.cuda())
                loss = loss + loss_direction_dice

        else:
            loss_direction_CE = torch.tensor(0.0)
            loss_direction_dice = torch.tensor(0.0)

        if (opt.model['mseloss'] == 1):

            if (len(target_point.shape) == 3):
                target_point = target_point.unsqueeze(1)
            loss_mse = criterion_MSE(output_point.cuda(), (target_point / 255).cuda())
            loss = loss + loss_mse

        else:
            loss_mse = torch.tensor(0.0)

        loss = loss.float()

        if (accuracy_tensor == 0):
            pred = np.argmax(log_prob_maps.data.cpu().numpy(), axis=1)
            metrics = utils.accuracy_pixel_level(pred, target.numpy())
        else:
            pred_tensor = torch.argmax(log_prob_maps, dim=1, keepdim=False)
            metrics = utils.accuracy_pixel_level_tensor(pred_tensor, target)

        pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1 = metrics[0], metrics[1], metrics[2], metrics[3], \
                                                                         metrics[4]

        # do_object_metric = 1
        if (do_object_metric == 1):
            start_time_do_object_metric = time.time()

            pred_inside = pred[0, :, :] == 1
            pred_inside2 = ndi.binary_fill_holes(pred_inside)
            pred2 = morph.remove_small_objects(pred_inside2, opt.post['min_area'])
            # measure.label 实例化；
            pred_labeled = measure.label(pred2)  # connected component labeling
            pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(opt.post['radius']))
            label_inside = ((target[0, :, :]).numpy() == 1)  # .astype(np.int8)
            # pred_labeled = morph.dilation(label_inside, selem=morph.selem.disk(1))
            label_img = label_inside.astype(np.uint8) * 255
            result_object = utils.nuclei_accuracy_object_level(pred_labeled, label_img)
            recall, precision, F1, dice, iou, haus, AJI = result_object[0], result_object[1], result_object[2], \
                                                          result_object[3], result_object[4], result_object[5], \
                                                          result_object[6]
            # print('recall = {}, precision = {}, F1 = {}, dice = {}, iou = {}, haus = {}, AJI = {}'.format(recall, precision, F1, dice, iou, haus, AJI))

        else:
            recall, precision, F1, dice, iou, haus, AJI = 0, 0, 0, 0, 0, 0, 0
            # val_obj_iou
            iou = pixel_iou

        results.update([loss.item(), loss_direction_CE.item(), loss_direction_dice.item(), loss_mse.item(),
                        pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1,
                        recall, precision, F1, dice, iou, haus, AJI,
                        ])

        del output, target_var, log_prob_maps, loss

    logger.info('\t=> Val Avg:   \tLoss {r[0]:.4f}'
                '\tloss_direction_CE {r[1]:.4f}'
                '\tloss_direction_dice {r[2]:.4f}'
                '\tloss_mse {r[3]:.4f}'
                '\tPixel_Acc {r[4]:.4f}'
                '\tPixel_IoU {r[5]:.4f}'
                '\tpixel_Recall {r[6]:.4f}'
                '\tpixel_Precision {r[7]:.4f}'
                '\tpixel_F1 {r[8]:.4f}'
                '\n\t\t obj_recall {r[9]:.4f}'
                '\tobj_precision {r[10]:.4f}'
                '\tobj_F1 {r[11]:.4f}'
                '\tobj_dice {r[12]:.4f}'
                '\tobj_iou {r[13]:.4f}'
                '\tobj_haus {r[14]:.4f}'
                '\tobj_AJI {r[15]:.4f}'.format(r=results.avg))

    return results.avg





























