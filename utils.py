import numpy as np
import math
import random
import torch
import skimage.morphology as morph
from scipy.spatial.distance import directed_hausdorff as hausdorff
from scipy import ndimage
from skimage import measure
import time


def accuracy_pixel_level_tensor(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]
    results = np.zeros((6,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics_tensor(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value / batch_size for value in results]


def compute_pixel_level_metrics_tensor(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """

    if not isinstance(pred, torch.Tensor):
        pred = torch.from_numpy(pred.astype(np.uint8)).float()   # pred.cuda() # .astype(np.uint8)
    if not isinstance(target, torch.Tensor):
        target = torch.from_numpy(target.astype(np.uint8)).float() # target.cuda() # .astype(np.uint8)

    pred = pred.cuda()
    target = target.cuda()
    tp = torch.sum(torch.mul(pred, target))  # true postives
    tn = torch.sum(torch.mul((1 - pred), (1 - target)))  # true negatives
    fp = torch.sum(torch.mul(pred, (1 - target)))  # false postives
    fn = torch.sum(torch.mul((1 - pred), target))  # false negatives

    precision = torch.div(tp.float(), (tp + fp + 1e-10).float())
    recall = torch.div(tp.float(), (tp + fn + 1e-10).float())
    F1 = torch.div(2 * torch.mul(precision, recall).float(), (precision + recall + 1e-10).float())
    acc = torch.div((tp + tn).float(), (tp + fp + tn + fn + 1e-10).float())
    performance = torch.div(recall + torch.div(tn.float(), (tn + fp + 1e-10)).float(), 2.0)
    iou = torch.div(tp.float(), (tp + fp + fn + 1e-10).float())

    precision = precision.item()  # data.cpu().numpy().max()
    recall = recall.item()
    F1 = F1.item()
    acc = acc.item()
    performance = performance.item()
    iou = iou.item()

    return [acc, iou, recall, precision, F1, performance]




def accuracy_pixel_level(output, target):
    """ Computes the accuracy during training and validation for ternary label """
    batch_size = target.shape[0]
    results = np.zeros((6,), np.float)

    for i in range(batch_size):
        pred = output[i, :, :]
        label = target[i, :, :]

        # inside part
        pred_inside = pred == 1
        label_inside = label == 1
        metrics_inside = compute_pixel_level_metrics(pred_inside, label_inside)

        results += np.array(metrics_inside)

    return [value / batch_size for value in results]



def compute_pixel_level_metrics(pred, target):
    """ Compute the pixel-level tp, fp, tn, fn between
    predicted img and groundtruth target
    """
    start_time = time.time()
    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(target, np.ndarray):
        target = np.array(target)

    tp = np.sum(pred * target)  # true postives
    tn = np.sum((1 - pred) * (1 - target))  # true negatives
    fp = np.sum(pred * (1 - target))  # false postives
    fn = np.sum((1 - pred) * target)  # false negatives

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    F1 = 2 * precision * recall / (precision + recall + 1e-10)
    acc = (tp + tn) / (tp + fp + tn + fn + 1e-10)
    performance = (recall + tn / (tn + fp + 1e-10)) / 2
    iou = tp / (tp + fp + fn + 1e-10)

    return [acc, iou, recall, precision, F1, performance]

from lxml import etree
def read_annotation(path):
    """
    [Summary]
        Read annotation file into buffer and parse it.
    [Arguments]
        path {pathlib.PosixPath}
        -- path to the annotation file.
    [Returns]:
        annotations {list}
        -- list of annotations with each annotation encoded
           as numpy.ndarray values.
    """
    tree = etree.parse(path)
    regions = tree.xpath("/Annotations/Annotation/Regions/Region")
    annotations = []
    for region in regions:
        points = []
        for point in region.xpath("Vertices/Vertex"):
            points.append([math.floor(float(point.attrib["X"])),
                           math.floor(float(point.attrib["Y"]))])
        annotations.append(np.array(points, dtype=np.int32))
    return annotations

import cv2
def to_mask_instance(annotations, height, width):
    """
    [Summary]
        Make the mask image from the annotation and image sizes.
    [Arguments]
        # Described as above.
    Returns:
        mask {numpy.ndarray}
        -- mask image with each pixels {0, 1}.
    """

    mask = np.zeros([height, width], dtype=np.float)
    mask = cv2.drawContours(mask, [annotations], 0, (1, 1, 1), thickness=cv2.FILLED)

    return mask

def nuclei_accuracy_annotation_object_level(pred, annotation_path):
    """ Computes the accuracy during test phase of nuclei segmentation """
    # get connected components
    height, width = pred.shape[0], pred.shape[1]

    annotations = read_annotation(annotation_path)
    annotations.sort(key=lambda i: len(i), reverse=True)
    annotations_len = len(annotations)
    Ng = annotations_len


    pred_labeled = measure.label(pred)
    #gt_labeled = measure.label(gt)
    Ns = len(np.unique(pred_labeled)) - 1  # number of detected objects
    #Ng = len(np.unique(gt_labeled)) - 1  # number of ground truth objects
    print('\tNs == %d, Ng == %d' % (Ns, Ng))
    TP = 0.0  # true positive
    FN = 0.0  # false negative
    dice = 0.0
    haus = 0.0
    iou = 0.0
    C = 0.0
    U = 0.0
    # pred_copy = np.copy(pred)
    count = 0.0

    for i in range(0, Ng): #for i in range(1, Ng + 1):
        annotation = annotations[i]
        gt_i = to_mask_instance(annotation, height, width)
        #gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_part = pred_labeled * gt_i

        # get intersection objects numbers in pred_labeled
        obj_no = np.unique(overlap_part)
        obj_no = obj_no[obj_no != 0]

        # no intersection object
        if obj_no.size == 0:
            FN += 1
            U += np.sum(gt_i)
            continue

        # find max iou object
        max_iou = 0.0
        for k in obj_no:
            tmp_overlap_area = np.sum(overlap_part == k)
            tmp_pred = np.where(pred_labeled == k, 1, 0)  # segmented object
            tmp_iou = float(tmp_overlap_area) / (np.sum(tmp_pred) + np.sum(gt_i) - tmp_overlap_area)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                pred_i = tmp_pred
                overlap_area = tmp_overlap_area

        TP += 1
        count += 1

        # compute dice and iou
        dice += 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
        iou += float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

        # compute hausdorff distance
        seg_ind = np.argwhere(pred_i)
        gt_ind = np.argwhere(gt_i)
        haus += max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        # compute AJI
        C += overlap_area
        U += np.sum(pred_i) + np.sum(gt_i) - overlap_area

        pred_labeled[pred_i > 0] = 0  # remove the used nucleus

    # compute recall, precision, F1
    FP = Ns - TP
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    if count == 0:
        count = 1
    else:
        print('\tCount = ', count)
    dice /= count
    iou /= count
    haus /= count

    # compute AJI
    U += np.sum(pred_labeled > 0)
    AJI = float(C) / U

    return recall, precision, F1, dice, iou, haus, AJI








def nuclei_accuracy_object_level(pred, gt):
    """ Computes the accuracy during test phase of nuclei segmentation """
    # get connected components
    pred_labeled = measure.label(pred)
    gt_labeled = measure.label(gt)
    Ns = len(np.unique(pred_labeled)) - 1  # number of detected objects
    Ng = len(np.unique(gt_labeled)) - 1  # number of ground truth objects
    print('\tNs == %d, Ng == %d' % (Ns, Ng))
    TP = 0.0  # true positive
    FN = 0.0  # false negative
    dice = 0.0
    haus = 0.0
    iou = 0.0
    C = 0.0
    U = 0.0
    # pred_copy = np.copy(pred)
    count = 0.0

    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_part = pred_labeled * gt_i

        # get intersection objects numbers in pred_labeled
        obj_no = np.unique(overlap_part)
        obj_no = obj_no[obj_no != 0]

        # no intersection object
        if obj_no.size == 0:
            FN += 1
            U += np.sum(gt_i)
            continue

        # find max iou object
        max_iou = 0.0
        for k in obj_no:
            tmp_overlap_area = np.sum(overlap_part == k)
            tmp_pred = np.where(pred_labeled == k, 1, 0)  # segmented object
            tmp_iou = float(tmp_overlap_area) / (np.sum(tmp_pred) + np.sum(gt_i) - tmp_overlap_area)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                pred_i = tmp_pred
                overlap_area = tmp_overlap_area

        TP += 1
        count += 1

        # compute dice and iou
        dice += 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
        iou += float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

        # compute hausdorff distance
        seg_ind = np.argwhere(pred_i)
        gt_ind = np.argwhere(gt_i)
        haus += max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        # compute AJI
        C += overlap_area
        U += np.sum(pred_i) + np.sum(gt_i) - overlap_area

        # pred_copy[pred_i > 0] = 0
        pred_labeled[pred_i > 0] = 0  # remove the used nucleus

    # compute recall, precision, F1
    FP = Ns - TP
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    if count == 0:
        count = 1
    else:
        print('\tCount = ', count)
    dice /= count
    iou /= count
    haus /= count

    # compute AJI
    U += np.sum(pred_labeled > 0)
    AJI = float(C) / U

    return recall, precision, F1, dice, iou, haus, AJI





def nuclei_object_level_list(pred, gt):
    """ Computes the accuracy during test phase of nuclei segmentation """
    # get connected components
    pred_labeled = measure.label(pred)
    gt_labeled = measure.label(gt)
    Ns = len(np.unique(pred_labeled)) - 1  # number of detected objects
    Ng = len(np.unique(gt_labeled)) - 1  # number of ground truth objects
    print('\tNs == %d, Ng == %d' % (Ns, Ng))

    # 记录 有预测到的实例，记为1；否则为0
    acc_ins_list = []
    # 记录 预测到的实例的面积大小(overlap_area)
    acc_area_list = []

    TP = 0.0  # true positive
    FN = 0.0  # false negative
    dice = 0.0
    iou = 0.0
    C = 0.0
    U = 0.0
    # pred_copy = np.copy(pred)
    count = 0.0

    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_part = pred_labeled * gt_i

        # get intersection objects numbers in pred_labeled
        obj_no = np.unique(overlap_part)
        obj_no = obj_no[obj_no != 0]

        # no intersection object
        if obj_no.size == 0:
            FN += 1
            U += np.sum(gt_i)
            acc_ins_list.append(0)
            acc_area_list.append(0)
            continue

        # find max iou object
        max_iou = 0.0
        for k in obj_no:
            tmp_overlap_area = np.sum(overlap_part == k)
            tmp_pred = np.where(pred_labeled == k, 1, 0)  # segmented object
            tmp_iou = float(tmp_overlap_area) / (np.sum(tmp_pred) + np.sum(gt_i) - tmp_overlap_area)
            if tmp_iou > max_iou:
                max_iou = tmp_iou
                pred_i = tmp_pred
                overlap_area = tmp_overlap_area

        TP += 1
        count += 1

        # compute dice and iou
        dice += 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
        iou += float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

        # compute AJI
        C += overlap_area
        U += np.sum(pred_i) + np.sum(gt_i) - overlap_area

        # pred_copy[pred_i > 0] = 0
        pred_labeled[pred_i > 0] = 0  # remove the used nucleus

        acc_ins_list.append(1)
        acc_area_list.append(overlap_area)


    # compute recall, precision, F1
    FP = Ns - TP
    recall = TP / (TP + FN + 1e-10)
    precision = TP / (TP + FP + 1e-10)
    F1 = 2 * TP / (2 * TP + FP + FN + 1e-10)

    if count == 0:
        count = 1
    else:
        print('\tCount = ', count)
    dice /= count
    iou /= count

    # compute AJI
    U += np.sum(pred_labeled > 0)
    AJI = float(C) / U

    #return recall, precision, F1, dice, iou, AJI
    return acc_ins_list, acc_area_list, recall, precision, F1, dice, iou, AJI










def gland_accuracy_object_level(pred, gt):
    """ Compute the object-level hausdorff distance between predicted  and
    groundtruth """

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(gt, np.ndarray):
        gt = np.array(gt)

    # get connected components
    pred_labeled = morph.label(pred, connectivity=2)
    Ns = len(np.unique(pred_labeled)) - 1
    gt_labeled = morph.label(gt, connectivity=2)
    gt_labeled = morph.remove_small_objects(gt_labeled, 3)  # remove 1 or 2 pixel noise in the image
    gt_labeled = morph.label(gt_labeled, connectivity=2)
    Ng = len(np.unique(gt_labeled)) - 1

    # show_figures((pred_labeled, gt_labeled))

    # --- compute F1 --- #
    TP = 0.0  # true positive
    FP = 0.0  # false positive
    for i in range(1, Ns + 1):
        pred_i = np.where(pred_labeled == i, 1, 0)
        img_and = np.logical_and(gt_labeled, pred_i)

        # get intersection objects in target
        overlap_parts = img_and * gt_labeled
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((img_i, overlap_parts))

        # no intersection object
        if obj_no.size == 0:
            FP += 1
            continue

        # find max overlap object
        obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
        gt_obj = obj_no[np.argmax(obj_areas)]  # ground truth object number

        gt_obj_area = np.sum(gt_labeled == gt_obj)  # ground truth object area
        overlap_area = np.sum(overlap_parts == gt_obj)

        if float(overlap_area) / gt_obj_area >= 0.5:
            TP += 1
        else:
            FP += 1

    FN = Ng - TP  # false negative

    if TP == 0:
        precision = 0
        recall = 0
        F1 = 0
    else:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)

    # --- compute dice, iou, hausdorff --- #
    pred_objs_area = np.sum(pred_labeled > 0)  # total area of objects in image
    gt_objs_area = np.sum(gt_labeled > 0)  # total area of objects in groundtruth gt

    # compute how well groundtruth object overlaps its segmented object
    dice_g = 0.0
    iou_g = 0.0
    hausdorff_g = 0.0
    for i in range(1, Ng + 1):
        gt_i = np.where(gt_labeled == i, 1, 0)
        overlap_parts = gt_i * pred_labeled

        # get intersection objects numbers in image
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        gamma_i = float(np.sum(gt_i)) / gt_objs_area

        # show_figures((pred_labeled, gt_i, overlap_parts))

        if obj_no.size == 0:  # no intersection object
            dice_i = 0
            iou_i = 0

            # find nearest segmented object in hausdorff distance
            min_haus = 1e5
            for j in range(1, Ns + 1):
                pred_j = np.where(pred_labeled == j, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                if haus_tmp < min_haus:
                    min_haus = haus_tmp
            haus_i = min_haus
        else:
            # find max overlap object
            obj_areas = [np.sum(overlap_parts == k) for k in obj_no]
            seg_obj = obj_no[np.argmax(obj_areas)]  # segmented object number
            pred_i = np.where(pred_labeled == seg_obj, 1, 0)  # segmented object

            overlap_area = np.max(obj_areas)  # overlap area

            dice_i = 2 * float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i))
            iou_i = float(overlap_area) / (np.sum(pred_i) + np.sum(gt_i) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_i)
            gt_ind = np.argwhere(gt_i)
            haus_i = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_g += gamma_i * dice_i
        iou_g += gamma_i * iou_i
        hausdorff_g += gamma_i * haus_i

    # compute how well segmented object overlaps its groundtruth object
    dice_s = 0.0
    iou_s = 0.0
    hausdorff_s = 0.0
    for j in range(1, Ns + 1):
        pred_j = np.where(pred_labeled == j, 1, 0)
        overlap_parts = pred_j * gt_labeled

        # get intersection objects number in gt
        obj_no = np.unique(overlap_parts)
        obj_no = obj_no[obj_no != 0]

        # show_figures((pred_j, gt_labeled, overlap_parts))

        sigma_j = float(np.sum(pred_j)) / pred_objs_area
        # no intersection object
        if obj_no.size == 0:
            dice_j = 0
            iou_j = 0

            # find nearest groundtruth object in hausdorff distance
            min_haus = 1e5
            for i in range(1, Ng + 1):
                gt_i = np.where(gt_labeled == i, 1, 0)
                seg_ind = np.argwhere(pred_j)
                gt_ind = np.argwhere(gt_i)
                haus_tmp = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

                if haus_tmp < min_haus:
                    min_haus = haus_tmp
            haus_j = min_haus
        else:
            # find max overlap gt
            gt_areas = [np.sum(overlap_parts == k) for k in obj_no]
            gt_obj = obj_no[np.argmax(gt_areas)]  # groundtruth object number
            gt_j = np.where(gt_labeled == gt_obj, 1, 0)  # groundtruth object

            overlap_area = np.max(gt_areas)  # overlap area

            dice_j = 2 * float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j))
            iou_j = float(overlap_area) / (np.sum(pred_j) + np.sum(gt_j) - overlap_area)

            # compute hausdorff distance
            seg_ind = np.argwhere(pred_j)
            gt_ind = np.argwhere(gt_j)
            haus_j = max(hausdorff(seg_ind, gt_ind)[0], hausdorff(gt_ind, seg_ind)[0])

        dice_s += sigma_j * dice_j
        iou_s += sigma_j * iou_j
        hausdorff_s += sigma_j * haus_j

    return recall, precision, F1, (dice_g + dice_s) / 2, (iou_g + iou_s) / 2, (hausdorff_g + hausdorff_s) / 2


def split_forward(model, input, size, overlap, opt):
    '''
    split the input image for forward process
    '''
    outchannel = opt.model['out_c']
    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    if opt.model['modelName'] == 'BRPNet':
        output = torch.zeros((input.size(0), outchannel*2, h, w))
    else:
        output = torch.zeros((input.size(0), outchannel, h, w))

    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                if opt.model['modelName'] == 'PraNet' or opt.model['modelName'] == 'SINet_V2':
                    output_patch, output4, output3, output2 = model(input_var)
                elif opt.model['modelName'] == 'BRPNet':
                    sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = model(input.cuda())
                    #output_patch = sout
                    output_patch = torch.cat((sout, cout), dim=1)
                else:
                    output_patch = model(input_var)

            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w
            output[:, :,ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :,ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]

    output = output[:, :, :h0, :w0].cuda()

    return output



def split_forward_dam(model, input, size, overlap, opt = None):
    '''
    split the input image for forward process
    '''

    b, c, h0, w0 = input.size()

    # zero pad for border patches
    pad_h = 0
    if h0 - size > 0:
        pad_h = (size - overlap) - (h0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, pad_h, w0))
        input = torch.cat((input, tmp), dim=2)

    if w0 - size > 0:
        pad_w = (size - overlap) - (w0 - size) % (size - overlap)
        tmp = torch.zeros((b, c, h0 + pad_h, pad_w))
        input = torch.cat((input, tmp), dim=3)

    _, c, h, w = input.size()

    output = torch.zeros((input.size(0), opt.model['out_c'], h, w))
    out_point_channel = 1
    output_point = torch.zeros((input.size(0), out_point_channel, h, w))
    output_direction = torch.zeros((input.size(0), opt.direction_classes, h, w))
    for i in range(0, h - overlap, size - overlap):
        r_end = i + size if i + size < h else h
        ind1_s = i + overlap // 2 if i > 0 else 0
        ind1_e = i + size - overlap // 2 if i + size < h else h
        for j in range(0, w - overlap, size - overlap):
            c_end = j + size if j + size < w else w

            input_patch = input[:, :, i:r_end, j:c_end]
            input_var = input_patch.cuda()
            with torch.no_grad():
                output_patch_all = model(input_var)
                if (len(output_patch_all) == 3):
                    output_patch = output_patch_all[0]
                    output_point_patch = output_patch_all[1]
                    output_direction_patch = output_patch_all[2]
                    
                elif (len(output_patch_all) == 2):
                    output_patch = output_patch_all[0]
                    if(opt.model['mseloss'] == 1):
                        output_point_patch = output_patch_all[1]
                    if(opt.model['direction'] == 1):
                        output_direction_patch = output_patch_all[1]
                elif (len(output_patch_all) == 1):
                    output_patch = output_patch_all


            ind2_s = j + overlap // 2 if j > 0 else 0
            ind2_e = j + size - overlap // 2 if j + size < w else w
            output[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_patch[:, :, ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]
            if(opt.model['mseloss'] == 1):
                output_point[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_point_patch[:, :, ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]
            if(opt.model['direction'] == 1):
                output_direction[:, :, ind1_s:ind1_e, ind2_s:ind2_e] = output_direction_patch[:, :, ind1_s - i:ind1_e - i, ind2_s - j:ind2_e - j]



    output = output[:, :, :h0, :w0].cuda()
    if(opt.model['mseloss'] == 1):
        output_point = output_point[:, :, :h0, :w0].cuda()
    if (opt.model['direction'] == 1):
        output_direction = output_direction[:, :, :h0, :w0].cuda()
        return output, output_point, output_direction
    else:
        return output, output_point







def get_random_color():
    ''' generate rgb using a list comprehension '''
    r, g, b = [random.random()*255 for i in range(3)]
    return r, g, b


def show_figures(imgs, new_flag=False):
    import matplotlib.pyplot as plt
    if new_flag:
        for i in range(len(imgs)):
            plt.figure()
            plt.imshow(imgs[i])
    else:
        for i in range(len(imgs)):
            plt.figure(i + 1)
            plt.imshow(imgs[i])

    plt.show()


# revised on https://github.com/pytorch/examples/blob/master/imagenet/main.py#L139
class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, shape=1):
        self.shape = shape
        self.reset()

    def reset(self):
        self.val = np.zeros(self.shape)
        self.avg = np.zeros(self.shape)
        self.sum = np.zeros(self.shape)
        self.count = 0

    def update(self, val, n=1):
        val = np.array(val)
        assert val.shape == self.val.shape
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def get_eachClassLossRate(i, loss_map, target_var):
    loss_re = loss_map.detach().cpu().numpy().reshape(-1)
    target_re = target_var.detach().cpu().numpy().reshape(-1)
    target_unique = np.unique(target_re)

    loss_0 = 0
    loss_0_number = 0
    loss_1 = 0
    loss_1_number = 0
    loss_2 = 0
    loss_2_number = 0

    for k in range(len(loss_re)):
        if(target_re[k] == target_unique[0]):
            loss_0 = loss_0 + loss_re[k]
            loss_0_number = loss_0_number + 1

        if (target_re[k] == target_unique[1]):
            loss_1 = loss_1 + loss_re[k]
            loss_1_number = loss_1_number + 1

        if (target_re[k] == target_unique[2]):
            loss_2 = loss_2 + loss_re[k]
            loss_2_number = loss_2_number + 1

    loss_0_mean = loss_0 / loss_0_number
    loss_1_mean = loss_1 / loss_1_number
    loss_2_mean = loss_2 / loss_2_number
    print('i = {}, loss_0_mean = {:2f}, loss_0 = {:2f}, loss_0_number = {:d}'.format(i, loss_0_mean, loss_0, loss_0_number))
    print('i = {}, loss_1_mean = {:2f}, loss_1 = {:2f}, loss_1_number = {:d}'.format(i, loss_1_mean, loss_1, loss_1_number))
    print('i = {}, loss_2_mean = {:2f}, loss_2 = {:2f}, loss2 = {:d}'.format(i, loss_2_mean, loss_2, loss_2_number))

    return loss_0_mean, loss_1_mean, loss_2_mean





def chooseModel(opt):
    
    # ----- create model ----- #  add models
    if opt.model['modelName'] == 'UNet':
        from models.unet import UNet
        model = UNet(num_classes=opt.model['out_c'], in_channels=opt.model['in_c'])
    elif opt.model['modelName'] == 'UNet_vgg16': #
        from models.model_unet import Unet
        model = Unet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=opt.model['out_c']) #True
    elif opt.model['modelName'] == 'UNet_resnet101': #
        from models.model_unet import Unet
        model = Unet(backbone_name='resnet101', pretrained=True, encoder_freeze=False, classes=opt.model['out_c']) #True
    elif opt.model['modelName'] == 'UNet_resnet50': #
        from models.model_unet import Unet
        model = Unet(backbone_name='resnet50', pretrained=True, encoder_freeze=False, classes=opt.model['out_c'])

    
    elif opt.model['modelName'] == 'FullNet':
        from models.FullNet import FullNet
        model = FullNet(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                        growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                        dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                        compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
        
    elif opt.model['modelName'] == 'FCN_pooling':
        from models.FullNet import FCN_pooling
        model = FCN_pooling(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                            growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                            dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                            compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])



    elif opt.model['modelName'] == 'UNet2RevA1_vgg16':  #
        from models.dam.model_unet_rev1 import Unet
        model = Unet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=opt.model['out_c'])



    
    #
    elif opt.model['modelName'] == 'model_unet_MandD':  #
        from models.dam.model_unet_MandD import Unet
        model = Unet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=opt.model['out_c'])
    #
    elif opt.model['modelName'] == 'model_unet_MandD4':  #
        from models.dam.model_unet_MandD4 import Unet
        model = Unet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=opt.model['out_c'])
    
    #
    elif opt.model['modelName'] == 'model_unet_MandD16':  #
        from models.dam.model_unet_MandD16 import Unet
        model = Unet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=opt.model['out_c'])
    

    
    #
    elif opt.model['modelName'] == 'model_unet_MandDandP':  #
        from models.dam.model_unet_MandDandP import Unet
        model = Unet(backbone_name='vgg16_bn', pretrained=True, encoder_freeze=False, classes=opt.model['out_c'])
    

    
    #
    elif opt.model['modelName'] == 'HRNet18_rev1':
        from models.dam.seg_hrnet_rev1 import HighResolutionNet
        model = HighResolutionNet(opt)


    
    return model
    





# ========================================= get_optimizer =============================================================
# from https://github.com/fv316/MAP583/blob/master/toolbox/optimizers.py
from hhl_utils.radam import RAdam, RAdam_4step, AdamW
from hhl_utils.ranger import Ranger


EPS = 1e-7


def get_optim_parameters(model):
    for param in model.parameters():
        yield param


def get_optimizer(args, model):
    optimizer, scheduler = None, None
    optimizer_name = args.train['optimizer']
    if(optimizer_name.lower() == 'sgd'):
        optimizer = torch.optim.SGD(get_optim_parameters(model),
                                    lr=args.train['lr'],
                                    momentum=args.momentum,
                                    weight_decay=args.train['weight_decay'])
    elif(optimizer_name.lower() == 'adam'):
        optimizer = torch.optim.Adam(get_optim_parameters(model),
                                     lr=args.train['lr'], betas=(0.9, 0.99),
                                     weight_decay = args.train['weight_decay'])
    
    elif (optimizer_name.lower() == 'radam'):
        optimizer = RAdam(model.parameters(), lr=args.train['lr'], betas=(0.9, 0.99),
                          weight_decay=args.train['weight_decay'])
    elif (optimizer_name.lower() == 'radam4s'):
        update_all = False
        additional_four = False
        optimizer = RAdam_4step(model.parameters(), lr=args.train['lr'], betas=(0.9, 0.99),
                                weight_decay=args.train['weight_decay'], update_all=update_all,
                                additional_four=additional_four)
    elif (optimizer_name.lower() == 'adamw'):
        warmup = 4000
        optimizer = AdamW(model.parameters(), lr=args.train['lr'], betas=(0.9, 0.99),
                          weight_decay=args.train['weight_decay'], warmup=warmup)
    elif (optimizer_name.lower() == 'ranger'):
        optimizer = Ranger(model.parameters(), args.train['lr'], betas=(0.9, 0.99),
                           weight_decay=args.train['weight_decay'])
    
    
    else:
        raise 'Optimizer {} not available'.format(args.optimizer)

    if 'StepLR' == args.train['scheduler']:
        print(f' --- Setting lr scheduler to StepLR ---')
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.train['step'], gamma=args.train['lr_decay'])
    elif 'ExponentialLR' == args.train['scheduler']:
        print(f' --- Setting lr scheduler to ExponentialLR ---')
        #即每个epoch都衰减lr = lr * gamma,即进行指数衰减
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=args.train['lr_decay'])
    elif 'ReduceLROnPlateau' == args.train['scheduler']:
        print(f' --- Setting lr scheduler to ReduceLROnPlateau ---')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', factor=args.train['lr_decay'], patience=args.train['step'])
    elif 'CosineAnnealingWarmRestarts' == args.train['scheduler']:  #
        print(f' --- Setting lr scheduler to CosineAnnealingWarmRestarts ---')
        # T_0=30
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.train['step'], T_mult=2, eta_min=0)
    else:
        print("Scheduler {} not available".format(args.train['scheduler']))
        #raise f'Scheduler {args.scheduler} not available'

    return optimizer, scheduler


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every args.step epochs"""
    #lr = args.lr * (0.1 ** (epoch // args.step))
    #
    if(args.train['scheduler'] == 'None'):
        lr = args.train['lr'] * 1#(0.1 ** (epoch // args.step))
    else:
        lr = args.train['lr'] * (0.9 ** (epoch // args.train['step']))
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
   
def adjust_learning_rate2(args, optimizer, epoch):
    gamma=0.999
    for param_group in optimizer.param_groups:
        lr = args.train['lr']
        param_group['lr'] = lr*math.pow(gamma,epoch)
        print ("decay",param_group['lr'])

        




# Codes modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, epoch):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.update(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'===================== EarlyStopping counter: {self.counter} out of {self.patience} =====================')
            if self.counter >= self.patience and epoch >= 100:
                self.early_stop = True
        else:
            self.best_score = score
            self.update(val_loss)
            self.counter = 0

    def update(self, val_loss):
        '''Update when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss

# EarlyStopping：https://blog.csdn.net/weixin_40519315/article/details/104633238



from skimage import morphology, io, color, measure
class Reinhard_normalizer(object):
    """
    color normalization using Reinhard's method
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target):
        #target = self._standardize_brightness(target)
        means, stds = self._get_mean_std(target)
        self.target_means = means
        self.target_stds = stds

    def transform(self, I):
        #I = self._standardize_brightness(I)
        I1, I2, I3 = self._lab_split(I)
        means, stds = self._get_mean_std(I)
        norm1 = ((I1 - means[0]) * (self.target_stds[0] / stds[0])) + self.target_means[0]
        norm2 = ((I2 - means[1]) * (self.target_stds[1] / stds[1])) + self.target_means[1]
        norm3 = ((I3 - means[2]) * (self.target_stds[2] / stds[2])) + self.target_means[2]
        return self._merge_back(norm1, norm2, norm3)

    def _lab_split(self, I):
        """
        Convert from RGB uint8 to LAB and split into channels
        """
        I = color.rgb2lab(I)
        I1, I2, I3 = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        return I1, I2, I3

    def _merge_back(self, I1, I2, I3):
        """
        Take seperate LAB channels and merge back to give RGB uint8
        """
        I = np.stack((I1, I2, I3), axis=2)
        return (color.lab2rgb(I) * 255).astype(np.uint8)

    def _get_mean_std(self, I):
        """
        Get mean and standard deviation of each channel
        """
        I1, I2, I3 = self._lab_split(I)
        means = np.mean(I1), np.mean(I2), np.mean(I3)
        stds = np.std(I1), np.std(I2), np.std(I3)
        return means, stds

    def _standardize_brightness(self, I):
        p = np.percentile(I, 90)
        return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def style_transfer(ori_img_batch, style_img_bacrh, transfer_rate):
    batch_size = ori_img_batch.shape[0]
    batch_start = np.int(batch_size * (1-transfer_rate))
    for batch_i in range(batch_start, batch_size):
        normalizer = Reinhard_normalizer()
        normalizer.fit(style_img_bacrh[batch_i])
        io.imsave('./images/ori_{}.png'.format(batch_i), ori_img_batch[batch_i])
        normalized_img = normalizer.transform(ori_img_batch[batch_i])
        ori_img_batch[batch_i] = normalized_img
        io.imsave('./images/transfer_{}.png'.format(batch_i), ori_img_batch[batch_i])
        io.imsave('./images/style_{}.png'.format(batch_i), style_img_bacrh[batch_i])


    return ori_img_batch









from scipy import stats
# DcmVoting
def DcmVoting(seg_map):
    height, width = seg_map.shape[0], seg_map.shape[1]
    tta_number = seg_map.shape[2]
    numberListAll = [[1, 2, 3, 4, 5, 6, 7, 8], [5, 4, 3, 2, 1, 8, 7, 6], [1, 8, 7, 6, 5, 4, 3, 2],
                     [5, 6, 7, 8, 1, 2, 3, 4], [7, 8, 1, 2, 3, 4, 5, 6], [7, 6, 5, 4, 3, 2, 1, 8],
                     [3, 2, 1, 8, 7, 6, 5, 4], [3, 4, 5, 6, 7, 8, 1, 2]]

    DCM_tta = np.zeros((height, width, tta_number), np.uint8)

    for index_i, numberlist in enumerate(numberListAll):
        if (index_i != 0):
            index_item_list = []
            for item in numberlist:
                indexValue = item  # ori_numberlist.index(item) + 1
                index_item = (seg_map[:, :, index_i] == indexValue)
                index_item_list.append(index_item)
            DCM_temp = np.zeros((height, width), np.uint8)
            for index_j, index_item in enumerate(index_item_list):
                DCM_temp[index_item] = index_j + 1
            DCM_tta[:, :, index_i] = DCM_temp
        else:
            DCM_tta[:, :, 0] = seg_map[:, :, 0]

    pred_direction = stats.mode(DCM_tta, axis=2)[0]
    pred_direction = pred_direction[:,:,0]

    return pred_direction




def DcmVoting2(direct_map):
    trans = [[0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 5, 4, 3, 2, 1, 8, 7, 6], [0, 1, 8, 7, 6, 5, 4, 3, 2], [0, 5, 6, 7, 8, 1, 2, 3, 4], [0, 3, 4, 5, 6, 7, 8, 1, 2], [0, 7, 6, 5, 4, 3, 2, 1, 8], [0, 3, 2, 1, 8, 7, 6, 5, 4], [0, 7, 8, 1, 2, 3, 4, 5, 6]]
    h, w = direct_map.shape[0], direct_map.shape[1]
    direct_map = np.transpose(direct_map, (2, 0, 1))
    pred = np.zeros([h, w, 9], dtype=np.uint8)
    for i in range(8):
        for j in range(9):
            pred[direct_map[i] == j, trans[i][j]] += 1
    pred_direction = np.argmax(pred, axis = 2)
    return pred_direction















