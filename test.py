import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import skimage.morphology as morph
from skimage import io
from skimage import measure
from scipy import ndimage as ndi
from scipy import misc
import scipy.io as scio
import utils
import stats_utils
import time
from options import Options
from my_transforms import get_transforms
from sklearn.metrics import jaccard_score
import numpy as np
import pandas as pd
import copy
import cv2
import postproc_other

import os
#path = r'D:\EditSoftware\CDNet'
#os.chdir(path)

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def main():
    # accuracy_tensor
    global accuracy_tensor
    accuracy_tensor = 1

    start_all_time = time.time()
    opt = Options(isTrain=False)
    opt.parse()
    opt.save_options()
    opt.print_options()
    branch = opt.test['branch']

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.test['gpu'])

    global all_img_test
    all_img_test = opt.all_img_test

    img_dir = opt.test['img_dir']
    label_dir = opt.test['label_dir']
    annotation_dir = opt.test['annotation_dir']
    weight_map_dir = opt.test['weight_map_dir']
    save_dir = opt.test['save_dir']
    model_path = opt.test['model_path']
    save_flag = opt.test['save_flag']
    tta = opt.test['tta']
    groundtruth = opt.test['groundtruth']
    testfile = opt.test['filename']

    # 保存展示图片
    save_view_dir = './images'
    create_folder(save_view_dir)
    # 保存展示图片具体位置
    detail_name = opt.dataset + '_' + opt.model['modelName']
    save_view_detail_dir = os.path.join(save_view_dir, detail_name)
    create_folder(save_view_detail_dir)

    # check if it is needed to compute accuracies
    eval_flag = True if label_dir else False

    # data transforms
    test_transform = get_transforms(opt.transform['test'])

    gpu_str = ''
    for gpu_i in opt.test['gpu']:
        gpu_str = gpu_str + str(gpu_i)
    # log
    logExl_path = './experiments/' + opt.dataset + '/'
    logExl_name = '{}_logExl_gpu{}.csv'.format(opt.dataset, gpu_str)
    logExl_PathAndName = logExl_path + logExl_name

    if (os.path.exists(logExl_PathAndName) == False):
        print('{:s}_logExl.csv have not been exist ! ! !'.format(opt.dataset))
    else:
        print('{:s}_logExl.csv have been exist.'.format(opt.dataset))
        logExl = pd.read_csv(logExl_PathAndName)
        logExl_columns = logExl.columns
        logExl_number = logExl.shape[0]

    log_eachItem = logExl.loc[logExl_number - 1].tolist()

    model = utils.chooseModel(opt)

    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # ----- load trained model ----- #
    print("=> loading trained model")
    best_checkpoint = torch.load(model_path)
    model.load_state_dict(best_checkpoint['state_dict'], False)
    best_epoch = best_checkpoint['epoch']
    print("=> loaded model at epoch {}".format(best_epoch))
    model = model.module

    # switch to evaluate mode
    model.eval()
    counter = 0
    print("=> Test begins:")

    img_names = os.listdir(img_dir)
    # print("img_names => :", img_names)
    img_names = [img_name for img_name in img_names if img_name[-3:] == 'png']
    print("img_names changes => :", img_names)
    # pixel_accu, recall, precision, F1, dice, iou, haus, (AJI) + pixel_iou, pixel_recall, pixel_precision, pixel_F1
    num_metrics = 12  # num_metrics
    # hovernet==>stats_utils.py result_AJI, result_Dice
    num_metrics = num_metrics + 2
    num_metrics = num_metrics + 4
    num_metrics = num_metrics + 4
    avg_results = utils.AverageMeter(num_metrics)


    scale_size_notmatch = []

    all_results = dict()
    ji_value = 0
    if save_flag:
        create_folder(save_dir)

        strs = img_dir.split('/')


        prob_maps_folder = '{:s}/{:s}_prob_maps'.format(save_dir, strs[-1])
        seg_folder = '{:s}/{:s}_segmentation'.format(save_dir, strs[-1])
        create_folder(prob_maps_folder)
        create_folder(seg_folder)
    # print(img_names)
    for img_name in img_names:
        # load test image
        print('=> Processing image {:s}'.format(img_name))
        img_path = '{:s}/{:s}'.format(img_dir, img_name)
        img = Image.open(img_path)
        ori_h = img.size[1]
        ori_w = img.size[0]
        name = os.path.splitext(img_name)[0]
        if eval_flag:
            if opt.dataset == 'MultiOrgan':
                label_path = '{:s}/{:s}_label.png'.format(label_dir, name)  # mask
                annotation_path = '{:s}/{:s}.xml'.format(annotation_dir, name)
                if(opt.test['filename'] == 'CoNSeP_test'):
                    img_type = 'npy'
                else:
                    img_type = 'mat' # npy
                if(branch == 5 and img_type == 'npy'):
                    label_instance_path = '{:s}_ins/{:s}.npy'.format(label_dir, name)
                    label_img_instance = np.load(label_instance_path)[:,:,0]
                    print('label_img_instance.len = {}'.format(len(np.unique(label_img_instance))))
                elif(branch == 5 and img_type == 'mat'):
                    label_instance_path = '{:s}_ins/{:s}.mat'.format(label_dir, name)
                    label_img_instance = scio.loadmat(label_instance_path)['inst_map']
                    print('label_img_instance.len = {}'.format(len(np.unique(label_img_instance))))

            if opt.dataset == 'MoNuSeg_oridata':
                label_path = '{:s}/{:s}_label.png'.format(label_dir, name)  # mask
                annotation_path = '{:s}/{:s}.xml'.format(annotation_dir, name)
                if (branch == 5):
                    label_instance_path = '{:s}_ins/{:s}.npy'.format(label_dir, name)
                    label_img_instance = np.load(label_instance_path)
                    print('label_img_instance.len = {}'.format(len(np.unique(label_img_instance))))
            elif opt.dataset == 'CPM2017':
                label_path = '{:s}/{:s}_label.png'.format(label_dir, name)
                img_type = 'mat'  # npy mat
                if (branch == 5 and img_type == 'npy'):
                    label_instance_path = '{:s}_ins/{:s}_label.npy'.format(label_dir, name)
                    label_img_instance = np.load(label_instance_path)[:, :, 0]
                    print('label_img_instance.len = {}'.format(len(np.unique(label_img_instance))))
                elif (branch == 5 and img_type == 'mat'):
                    label_instance_path = '{:s}_ins/{:s}_label.mat'.format(label_dir, name)
                    label_img_instance = scio.loadmat(label_instance_path)['inst_map']
                    print('label_img_instance.len = {}'.format(len(np.unique(label_img_instance))))
            else:
                label_path = '{:s}/{:s}_label.png'.format(label_dir, name)

                img_type = 'npy'  # npy  mat
                if (branch == 5 and img_type == 'npy'):
                    label_instance_path = '{:s}_ins/{:s}.npy'.format(label_dir, name)
                    label_img_instance = np.load(label_instance_path)#[:, :, 0]
                    print('label_img_instance.shape = {}, label_img_instance.len = {}'.format(label_img_instance.shape, len(np.unique(label_img_instance))))
                elif (branch == 5 and img_type == 'mat'):
                    label_instance_path = '{:s}_ins/{:s}_label.mat'.format(label_dir, name)
                    label_img_instance = scio.loadmat(label_instance_path)['inst_map']
                    print('label_img_instance.shape = {}, label_img_instance.len = {}'.format(label_img_instance.shape, len(np.unique(label_img_instance))))
                    

            label_img = io.imread(label_path)
        
        label_ins_h = label_img_instance.shape[0]
        label_ins_w = label_img_instance.shape[1]
        
        if (ori_h != label_ins_h or ori_w !=label_ins_w):
            print('img.size = {}, label.shape = {}'.format(img.size, label_img_instance.shape))
            notmatch_item = []
            notmatch_item.append(img_name)
            notmatch_item.append(img.size)
            notmatch_item.append(label_img_instance.shape)
            scale_size_notmatch.append(notmatch_item)
            continue
        
        
        
        input = test_transform((img,))[0].unsqueeze(0)
        print('\tComputing output probability maps...')
        start_test_time = time.time()
        prob_maps = get_probmaps(input, model, opt, 1)

        if (counter < 10):
            end_test_time = time.time()
            work_test_time = end_test_time - start_test_time
            work_test_time = round(work_test_time, 2)
            work_test_time_m = work_test_time  # / 60
            print('===> Processed all {:d} images, ===> The every test time spent is [{:.2f} s]'.format(counter,
                                                                                                        work_test_time_m))

        # tta = 0
        if tta:
            img_hf = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_vf = img.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_hvf = img_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_hf = test_transform((img_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_vf = test_transform((img_vf,))[0].unsqueeze(0)  # vertical flip input
            input_hvf = test_transform((img_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_hf = get_probmaps(input_hf, model, opt, 1)
            prob_maps_vf = get_probmaps(input_vf, model, opt, 1)
            prob_maps_hvf = get_probmaps(input_hvf, model, opt, 1)

            # re flip
            prob_maps_hf = np.flip(prob_maps_hf, 2)
            prob_maps_vf = np.flip(prob_maps_vf, 1)
            prob_maps_hvf = np.flip(np.flip(prob_maps_hvf, 1), 2)

            # rotation 90 and flips
            img_r90 = img.rotate(90, expand=True)
            img_r90_hf = img_r90.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
            img_r90_vf = img_r90.transpose(Image.FLIP_TOP_BOTTOM)  # vertical flip
            img_r90_hvf = img_r90_hf.transpose(Image.FLIP_TOP_BOTTOM)  # horizontal and vertical flips

            input_r90 = test_transform((img_r90,))[0].unsqueeze(0)
            input_r90_hf = test_transform((img_r90_hf,))[0].unsqueeze(0)  # horizontal flip input
            input_r90_vf = test_transform((img_r90_vf,))[0].unsqueeze(0)  # vertical flip input
            input_r90_hvf = test_transform((img_r90_hvf,))[0].unsqueeze(0)  # horizontal and vertical flip input

            prob_maps_r90 = get_probmaps(input_r90, model, opt, 1)
            prob_maps_r90_hf = get_probmaps(input_r90_hf, model, opt, 1)
            prob_maps_r90_vf = get_probmaps(input_r90_vf, model, opt, 1)
            prob_maps_r90_hvf = get_probmaps(input_r90_hvf, model, opt, 1)

            # re flip
            prob_maps_r90 = np.rot90(prob_maps_r90, k=3, axes=(1, 2))
            prob_maps_r90_hf = np.rot90(np.flip(prob_maps_r90_hf, 2), k=3, axes=(1, 2))
            prob_maps_r90_vf = np.rot90(np.flip(prob_maps_r90_vf, 1), k=3, axes=(1, 2))
            prob_maps_r90_hvf = np.rot90(np.flip(np.flip(prob_maps_r90_hvf, 1), 2), k=3, axes=(1, 2))

            prob_maps = (prob_maps + prob_maps_hf + prob_maps_vf + prob_maps_hvf
                         + prob_maps_r90 + prob_maps_r90_hf + prob_maps_r90_vf + prob_maps_r90_hvf) / 8

        if (opt.model['multi_class'] == False):
            pred = prob_maps[0] >= 0.5
            pred_inside = pred
        else:
            pred = np.argmax(prob_maps, axis=0)
            pred_inside = pred == 1

        pred_inside2 = ndi.binary_fill_holes(pred_inside)

        pred2 = morph.remove_small_objects(pred_inside2, opt.post['min_area'])  # remove small object

        if 'scale' in opt.transform['test']:
            pred2 = misc.imresize(pred2.astype(np.uint8) * 255, (ori_h, ori_w), interp='bilinear')
            pred2 = (pred2 > 127.5)

        pred2 = pred2.astype(np.uint8)

        # 1：watershed postproproc； 0：measure.label()
        if (int(opt.post['postproc']) == 1):
            pred_labeled = postproc_other.process(pred_inside.astype(np.uint8) * 255, model_mode=opt.model['modelName'],
                                                  min_size=opt.post['min_area'])
        else:
            pred_labeled = measure.label(pred2)  # connected component labeling

        # morph.dilation
        pred_labeled = morph.dilation(pred_labeled, selem=morph.selem.disk(opt.post['radius']))
        pred_labeled2 = pred2.astype(np.uint8) * 255


        if (branch == 5):
            label_inside = label_img[:, :] > 0
            label_instance_img = copy.deepcopy(label_img_instance)
            label_img = (label_img_instance[:, :] > 0).astype(np.uint8) * 255
        else:
            label_inside = label_img[:, :, 0] > 0
            label_img = label_inside.astype(np.uint8) * 255
            label_img_instance = copy.deepcopy(label_img)


        ji1 = jaccard_score(pred_labeled2, label_img, average='samples')  # samples

        ji_value += ji1

        if (branch == 5):
            label_img = label_instance_img

        if eval_flag:
            print('\tComputing metrics...')
            # np.expand_dims(a, axis=0)  (1000, 1000)-->(1,1000,1000)
            if (accuracy_tensor == 0):
                result = utils.accuracy_pixel_level(np.expand_dims(pred_labeled > 0, 0),
                                                    np.expand_dims(label_img > 0, 0))
            else:

                result = utils.accuracy_pixel_level_tensor(np.expand_dims(pred_labeled > 0, 0),
                                                           np.expand_dims(label_img > 0, 0))

            pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1 = result[0], result[1], result[2], result[3], \
                                                                             result[4]

            if opt.dataset != 'ABC':
                if groundtruth == 1:
                    result_object = utils.nuclei_accuracy_annotation_object_level(pred_labeled,
                                                                                  annotation_path)
                else:
                    result_object = utils.nuclei_accuracy_object_level(pred_labeled, label_img)

                user_hovernet_utils = 1
                if (user_hovernet_utils == 1):
                    pred_labeled = measure.label(pred_labeled)
                    gt_labeled = measure.label(label_img)
                    #result_AJI = stats_utils.get_fast_aji(gt_labeled, pred_labeled)
                    result_AJI, analysis_FP, analysis_FN, analysis_pred_less, analysis_pred_more = stats_utils.get_fast_aji(gt_labeled, pred_labeled)
                    result_Dice = stats_utils.get_dice_1(gt_labeled, pred_labeled)
                    result_Dice2 = 0 #stats_utils.get_dice_2(gt_labeled, pred_labeled)  # time consuming
                    get_fast_pq_value = stats_utils.get_fast_pq(gt_labeled, pred_labeled, match_iou=0.5)
                    pq_info = get_fast_pq_value[0]
                    paired_value = get_fast_pq_value[1]
                    dq_value = pq_info[0]  # dq
                    sq_value = pq_info[1]  # sq
                    pq_value = pq_info[2]  # pq

            else:
                result_object = utils.gland_accuracy_object_level(pred_labeled, label_img)

            print(
                '\timage {:s}, the pixel_iou = {:.4f}, pixel_recall = {:.4f}, pixel_precision = {:.4f}, pixel_F1 = {:.4f}'.format(
                    img_name, pixel_iou, pixel_recall, pixel_precision, pixel_F1))
            # hovernet==>stats_utils.py result_AJI, result_Dice
            print('\t {}, result_AJI = {}, result_Dice = {}, result_Dice2 = {}, dq_value = {}, sq_value = {}, pq_value = {}'.format(
                result_object, result_AJI, result_Dice, result_Dice2, dq_value, sq_value, pq_value))

            # 202108 add new feature
            all_results[name] = tuple(
                [pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1, *result_object, result_AJI, result_Dice,
                 result_Dice2, dq_value, sq_value, pq_value,
                 analysis_FP, analysis_FN, analysis_pred_less, analysis_pred_more])

            avg_results.update(
                [pixel_accu, pixel_iou, pixel_recall, pixel_precision, pixel_F1, *result_object, result_AJI, result_Dice,
                 result_Dice2, dq_value, sq_value, pq_value,
                 analysis_FP, analysis_FN, analysis_pred_less, analysis_pred_more])

        if save_flag:
            print('\tSaving image results...')

            cv2.imwrite('{:s}/b{:s}_{:s}_prob_inside.png'.format(prob_maps_folder, str(branch), name),
                        prob_maps[1, :, :] * 255)

            final_pred = Image.fromarray(pred_labeled.astype(np.uint16))
            final_pred.save('{:s}/b{:s}_{:s}_seg.tiff'.format(seg_folder, str(branch), name))

            # save colored objects
            pred_colored = np.zeros((ori_h, ori_w, 3))
            pred_labeled_cnum = pred_labeled.max() + 1
            for k in range(1, pred_labeled_cnum):
                pred_colored[pred_labeled == k, :] = np.array(utils.get_random_color())
            filename = '{:s}/b{:s}_{:s}_seg_colored.png'.format(seg_folder, str(branch), name)
            print('pred_labeled color number = %d' % (pred_labeled_cnum))

            cv2.imwrite(filename, pred_colored)

            pred_pred_label_name = '{}/{:s}_pred_label.png'.format(save_view_detail_dir, name)

            cv2.imwrite(pred_pred_label_name, pred_colored)

        counter += 1
        if counter % 10 == 0:
            print('\tProcessed {:d} images'.format(counter))

    AJI_sklearn_mean = ji_value / counter
    

    print('scale_size_notmatch = {}'.format(scale_size_notmatch))
    
    end_all_time = time.time()
    work_all_time = end_all_time - start_all_time
    work_all_time = round(work_all_time, 2)
    work_minute_time = work_all_time / 60
    print('===> Processed all {:d} images, ===> The time spent is [{:.2f} min]'.format(counter, work_minute_time))

    if eval_flag:
        print('Average of all images:\n'
              '\t pixel_accu: {r[0]:.4f}'
              '\t pixel_IoU {r[1]:.4f}'
              '\t pixel_Recall {r[2]:.4f}'
              '\t pixel_Precision {r[3]:.4f}'
              '\t pixel_F1 {r[4]:.4f}'
              '\n\t recall: {r[5]:.4f}'
              '\t\t precision: {r[6]:.4f}'
              '\t F1: {r[7]:.4f}'
              '\t dice: {r[8]:.4f}'
              '\t iou: {r[9]:.4f}'
              '\t haus: {r[10]:.4f}'.format(r=avg_results.avg))
        if opt.dataset != 'ABC':
            print('\t AJI_sklearn: {:.4f}'.format(AJI_sklearn_mean))
            print('\n\t AJI: {r[11]:.4f}'
                  '\t hover_AJI: {r[12]:.4f}'
                  '\t hover_Dice: {r[13]:.4f}'.format(r=avg_results.avg))
            
            print('\t result_Dice2: {r[14]:.4f}, '
                  '\t dq_value: {r[15]:.4f}, '
                  '\t sq_value: {r[16]:.4f}, '
                  '\t pq_value: {r[17]:.4f}'
                  '\t analysis_FP: {r[18]:.4f}, '
                  '\t analysis_FN: {r[19]:.4f}, '
                  '\t analysis_pred_less: {r[20]:.4f}, '
                  '\t analysis_pred_more: {r[21]:.4f}'.format(r=avg_results.avg))
            


        strs = img_dir.split('/')
        header = ['pixel_acc', 'pixel_IoU', 'pixel_Recall', 'pixel_Precision', 'pixel_F1', 'recall', 'precision', 'F1',
                  'Dice', 'IoU', 'Hausdorff']
        if opt.dataset != 'ABC':
            header.append('AJI')
            header.append('AJI_h')
            header.append('Dice_h')
            header.append('result_Dice2')
            header.append('dq_value')
            header.append('sq_value')
            header.append('pq_value')
            header.append('Ana_FP')
            header.append('Ana_FN')
            header.append('Ana_less')
            header.append('Ana_more')

        save_results(header, avg_results.avg, all_results,
                     '{:s}/{:s}_result.txt'.format(save_dir, strs[-1]))
        # save to auto_saveLog.txt
        # save_logTxt = './experiments/' + dataset + '/auto_saveLog.txt'
        gpu_str = ''
        for gpu_i in opt.test['gpu']:
            gpu_str = gpu_str + str(gpu_i)
        save_logTxt = './experiments/' + opt.dataset + '/auto_saveLog_gpu' + gpu_str + '.txt'
        save_results(header, avg_results.avg, all_results, save_logTxt, mode="a+")

        res_n_start = 38

        if (opt.test['filename'] != 'test2'):
            if (log_eachItem[2] == 'Test_epoch'):
                log_eachItem[0] = opt.model['modelName']
                log_eachItem[2] = best_epoch
                log_eachItem[6] = 1 if '_3c' in opt.test['save_dir'] else 0
                log_eachItem[11] = opt.model['mseloss']
                log_eachItem[12] = opt.model['direction']
                log_eachItem[15] = opt.model['LossName']  # "_CE1_Dice1"
                log_eachItem[23] = opt.post['postproc']
                log_eachItem[24] = opt.post['min_area']
                log_eachItem[25] = opt.post['radius']
                log_eachItem[27] = opt.test['groundtruth']
                log_eachItem[28] = all_img_test
                log_eachItem[29] = 1 if '_isRRe' in opt.test['save_dir'] else 0
                log_eachItem[30] = 1 if '_isRCo' in opt.test['save_dir'] else 0
                log_eachItem[31] = 1 if '_isRA' in opt.test['save_dir'] else 0
                log_eachItem[32] = 1 if '_isHF' in opt.test['save_dir'] else 0
                log_eachItem[33] = 1 if '_isRE' in opt.test['save_dir'] else 0
                log_eachItem[34] = 1 if '_isRRo' in opt.test['save_dir'] else 0
                log_eachItem[35] = 1 if '_isCAu' in opt.test['save_dir'] else 0
                log_eachItem[36] = 1 if '_isRCr' in opt.test['save_dir'] else 0
                log_eachItem[37] = 1 if '_isNorm' in opt.test['save_dir'] else 0

                avg_res = avg_results.avg
                log_eachItem[res_n_start] = round(avg_res[0], 7)
                log_eachItem[res_n_start + 1] = round(avg_res[1], 7)
                log_eachItem[res_n_start + 2] = round(avg_res[2], 7)
                log_eachItem[res_n_start + 3] = round(avg_res[3], 7)
                log_eachItem[res_n_start + 4] = round(avg_res[4], 7)
                log_eachItem[res_n_start + 5] = round(avg_res[5], 7)
                log_eachItem[res_n_start + 6] = round(avg_res[6], 7)
                log_eachItem[res_n_start + 7] = round(avg_res[7], 7)
                log_eachItem[res_n_start + 8] = round(avg_res[8], 7)
                log_eachItem[res_n_start + 9] = round(avg_res[9], 7)
                log_eachItem[res_n_start + 10] = round(avg_res[10], 7)
                log_eachItem[res_n_start + 11] = round(avg_res[11], 7)
                log_eachItem[res_n_start + 12] = round(AJI_sklearn_mean, 7)  # t1_AJI_sklearn

                log_eachItem[res_n_start + 13] = round(avg_res[12], 7)
                log_eachItem[res_n_start + 14] = round(avg_res[13], 7)

                log_eachItem[res_n_start + 15] = round(avg_res[14], 7)
                log_eachItem[res_n_start + 16] = round(avg_res[15], 7)
                log_eachItem[res_n_start + 17] = round(avg_res[16], 7)
                log_eachItem[res_n_start + 18] = round(avg_res[17], 7)
                log_eachItem[res_n_start + 19] = round(avg_res[18], 7)
                log_eachItem[res_n_start + 20] = round(avg_res[19], 7)
                log_eachItem[res_n_start + 21] = round(avg_res[20], 7)
                log_eachItem[res_n_start + 22] = round(avg_res[21], 7)
                logExl.loc[logExl_number - 1] = log_eachItem
                logExl.to_csv(logExl_path + logExl_name, index=False)
            else:
                log_eachItem = logExl.loc[logExl_number - 1].tolist()
                log_eachItem[0] = opt.model['modelName']
                log_eachItem[2] = best_epoch
                log_eachItem[6] = 1 if '_3c' in opt.test['save_dir'] else 0
                log_eachItem[11] = opt.model['mseloss']
                log_eachItem[12] = opt.model['direction']
                log_eachItem[15] = opt.model['LossName']  # "_CE1_Dice1"
                log_eachItem[23] = opt.post['postproc']
                log_eachItem[24] = opt.post['min_area']
                log_eachItem[25] = opt.post['radius']
                log_eachItem[27] = opt.test['groundtruth']
                log_eachItem[28] = all_img_test
                log_eachItem[29] = 1 if '_isRRe' in opt.test['save_dir'] else 0
                log_eachItem[30] = 1 if '_isRCo' in opt.test['save_dir'] else 0
                log_eachItem[31] = 1 if '_isRA' in opt.test['save_dir'] else 0
                log_eachItem[32] = 1 if '_isHF' in opt.test['save_dir'] else 0
                log_eachItem[33] = 1 if '_isRE' in opt.test['save_dir'] else 0
                log_eachItem[34] = 1 if '_isRRo' in opt.test['save_dir'] else 0
                log_eachItem[35] = 1 if '_isCAu' in opt.test['save_dir'] else 0
                log_eachItem[36] = 1 if '_isRCr' in opt.test['save_dir'] else 0
                log_eachItem[37] = 1 if '_isNorm' in opt.test['save_dir'] else 0

                avg_res = avg_results.avg
                log_eachItem[res_n_start] = round(avg_res[0], 7)
                log_eachItem[res_n_start + 1] = round(avg_res[1], 7)
                log_eachItem[res_n_start + 2] = round(avg_res[2], 7)
                log_eachItem[res_n_start + 3] = round(avg_res[3], 7)
                log_eachItem[res_n_start + 4] = round(avg_res[4], 7)
                log_eachItem[res_n_start + 5] = round(avg_res[5], 7)
                log_eachItem[res_n_start + 6] = round(avg_res[6], 7)
                log_eachItem[res_n_start + 7] = round(avg_res[7], 7)
                log_eachItem[res_n_start + 8] = round(avg_res[8], 7)
                log_eachItem[res_n_start + 9] = round(avg_res[9], 7)
                log_eachItem[res_n_start + 10] = round(avg_res[10], 7)
                log_eachItem[res_n_start + 11] = round(avg_res[11], 7)
                log_eachItem[res_n_start + 12] = round(AJI_sklearn_mean, 7)  # t1_AJI_sklearn

                log_eachItem[res_n_start + 13] = round(avg_res[12], 7)
                log_eachItem[res_n_start + 14] = round(avg_res[13], 7)

                log_eachItem[res_n_start + 15] = round(avg_res[14], 7)
                log_eachItem[res_n_start + 16] = round(avg_res[15], 7)
                log_eachItem[res_n_start + 17] = round(avg_res[16], 7)
                log_eachItem[res_n_start + 18] = round(avg_res[17], 7)
                log_eachItem[res_n_start + 19] = round(avg_res[18], 7)
                log_eachItem[res_n_start + 20] = round(avg_res[19], 7)
                log_eachItem[res_n_start + 21] = round(avg_res[20], 7)
                log_eachItem[res_n_start + 22] = round(avg_res[21], 7)
                logExl.loc[logExl_number] = log_eachItem
                logExl.to_csv(logExl_path + logExl_name, index=False)

        else:

            if (log_eachItem[2] == 'Test_epoch'):
                log_eachItem[2] = opt.test['epoch']
            avg_res = avg_results.avg

            m_len = len(avg_res) + 1
            log_eachItem[res_n_start + m_len] = round(avg_res[0], 7)
            log_eachItem[res_n_start + m_len + 1] = round(avg_res[1], 7)
            log_eachItem[res_n_start + m_len + 2] = round(avg_res[2], 7)
            log_eachItem[res_n_start + m_len + 3] = round(avg_res[3], 7)
            log_eachItem[res_n_start + m_len + 4] = round(avg_res[4], 7)
            log_eachItem[res_n_start + m_len + 5] = round(avg_res[5], 7)
            log_eachItem[res_n_start + m_len + 6] = round(avg_res[6], 7)
            log_eachItem[res_n_start + m_len + 7] = round(avg_res[7], 7)
            log_eachItem[res_n_start + m_len + 8] = round(avg_res[8], 7)
            log_eachItem[res_n_start + m_len + 9] = round(avg_res[9], 7)
            log_eachItem[res_n_start + m_len + 10] = round(avg_res[10], 7)
            log_eachItem[res_n_start + m_len + 11] = round(avg_res[11], 7)
            log_eachItem[res_n_start + m_len + 12] = round(AJI_sklearn_mean, 7)  # t1_AJI_sklearn

            log_eachItem[res_n_start + m_len + 13] = round(avg_res[12], 7)
            log_eachItem[res_n_start + m_len + 14] = round(avg_res[13], 7)

            log_eachItem[res_n_start + m_len + 15] = round(avg_res[14], 7)
            log_eachItem[res_n_start + m_len + 16] = round(avg_res[15], 7)
            log_eachItem[res_n_start + m_len + 17] = round(avg_res[16], 7)
            log_eachItem[res_n_start + m_len + 18] = round(avg_res[17], 7)
            log_eachItem[res_n_start + m_len + 19] = round(avg_res[18], 7)
            log_eachItem[res_n_start + m_len + 20] = round(avg_res[19], 7)
            log_eachItem[res_n_start + m_len + 21] = round(avg_res[20], 7)
            log_eachItem[res_n_start + m_len + 22] = round(avg_res[21], 7)
            
            logExl.loc[logExl_number - 1] = log_eachItem
            logExl.to_csv(logExl_path + logExl_name, index=False)


def get_probmaps(input, model, opt, branch_value):
    if (all_img_test == 1):
        size = 0  # 0 全图预测
    else:
        size = opt.test['patch_size']
    overlap = opt.test['overlap']

    if size == 0:
        with torch.no_grad():
            if opt.model['modelName'] == 'PraNet' or opt.model['modelName'] == 'SINet_V2':
                output, output4, output3, output2 = model(input.cuda())
            elif opt.model['modelName'] == 'BRPNet':
                sout, sout_0, sout_1, sout_2, sout_3, cout, cout_0, cout_1, cout_2, cout_3 = model(input.cuda())
                #output = sout
                output = torch.cat((sout, cout), dim=1)
            else:
                output = model(input.cuda())
    else:
        output = utils.split_forward(model, input, size, overlap, opt)

    if (len(output) == 2):
        output = output[0].squeeze(0)
    else:
        output = output.squeeze(0)

    prob_maps = F.softmax(output, dim=0).cpu().numpy()

    return prob_maps


def save_results(header, avg_results, all_results, filename, mode='w'):
    """ Save the result of metrics
        results: a list of numbers
    """
    N = len(header)
    assert N == len(avg_results)
    with open(filename, mode) as file:
        # header
        file.write('Metrics:\t')
        for i in range(N - 1):
            file.write('{:s}\t'.format(header[i]))
        file.write('{:s}\n'.format(header[N - 1]))

        # average results
        file.write('Average:\t')
        for i in range(N - 1):
            file.write('{:.4f}\t'.format(avg_results[i]))
        file.write('{:.4f}\n'.format(avg_results[N - 1]))
        file.write('\n')

        # all results
        for key, values in sorted(all_results.items()):
            file.write('{:s}:'.format(key))
            for value in values:
                file.write('\t{:.4f}'.format(value))
            file.write('\n')


if __name__ == '__main__':
    main()
