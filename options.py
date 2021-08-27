# CDNet

import os
import numpy as np
import argparse
from collections import OrderedDict


# RRe: random_resize; RCo:random_color; HF: horizontal_flip; RA: random_affine; RE: random_elastic;
# RRo: random_rotation; RCr: random_crop; LE: label_encoding;
def get_transformString(trans_train):
    transform_str = ''
    transform_str = transform_str + '_isRRe' if 'random_resize' in trans_train else transform_str + '_noRRe'
    transform_str = transform_str + '_isRCo' if 'random_color' in trans_train else transform_str + '_noRCo'
    transform_str = transform_str + '_isHF' if 'horizontal_flip' in trans_train else transform_str + '_noHF'
    transform_str = transform_str + '_isRA' if 'random_affine' in trans_train else transform_str + '_noRA'
    transform_str = transform_str + '_isRE' if 'random_elastic' in trans_train else transform_str + '_noRE'
    transform_str = transform_str + '_isRRo' if 'random_rotation' in trans_train else transform_str + '_noRRo'

    transform_str = transform_str + '_isRCr' if 'random_crop' in trans_train else transform_str + '_noRCr'
    transform_str = transform_str + '_isCAu' if 'random_chooseAug' in trans_train else transform_str + '_noCAu'
    
    transform_str = transform_str + '_isLE' if 'label_encoding' in trans_train else transform_str + '_noLE'
    
    #transform_str = transform_str + '_toTensor' if ('to_tensor' in trans_train) else transform_str + '_noTensor'
    transform_str = transform_str + '_isNorm' if ('normalize' in trans_train) else transform_str + '_noNorm'
    
    return transform_str


class Options:
    def __init__(self, isTrain):
        self.dataset = 'MoNuSeg_oridata'
        self.isTrain = isTrain
        self.all_img_test = 1
        self.momentum = 0.95
        # self.index = '1'
        # --- model hyper-parameters --- #
        self.model = dict()
        self.model['multi_class'] = True
        self.model['in_c'] = 1 if self.dataset == 'BBBC039V1' else 3# input channel
        self.model['out_c'] = 3 if self.model['multi_class'] == True else 1   # output channel
        
        self.model['direction'] = 1  # 1:add direction; 0:without add direction
        self.direction_classes = 8 + 1  # 4 8 16

        self.model['n_layers'] = 6  # number of layers in a block
        self.model['growth_rate'] = 24  # growth_rate
        self.model['drop_rate'] = 0.1
        self.model['compress_ratio'] = 0.5

        #self.model['dilations'] = [1, 2, 4, 8, 16, 4, 1]
        self.model['is_hybrid'] = True
        self.model['layer_type'] = 'basic'
        self.model['mean_std'] = 'mean_std'  
        
        self.model['add_weightMap'] = 1  # True # False
        self.model['dice'] = 1  # use Dice loss
        self.model['boundary_loss'] = 0  # use boundary loss
        self.model['mseloss'] = 1  # use MSE loss

        # UNet_vgg16 model_unet_MandD  model_unet_MandD4 model_unet_MandD16  model_unet_MandDandP  UNet2RevA1_vgg16
        self.model['modelName'] = 'UNet2RevA1_vgg16'
        self.model['backbone'] = 'None'
        self.model['pretrained'] = 1
        self.model['LossName'] = "CE1_Dice1"
        

        # --- training params --- #
        self.train = dict()
        self.train['branch'] = 5  # branch
        
        self.train['num_epochs'] = 300  # number of training epochs
        self.train['input_size'] = 256  # input size of the image
        self.train['batch_size'] = 8  # batch size
        self.train['val_overlap'] = 40  # overlap size of patches for validation
        
        self.train['seed'] = 2022
        self.train['early_stop'] = 7
        self.train['scheduler'] = 'None'
        self.train['step'] = 5
        self.train['lr'] = 0.001  # initial learning rate
        self.train['lr_decay'] = 0.995
        self.train['weight_decay'] = 1e-4  # weight decay
        
        
        self.train['log_interval'] = 15  # iterations to print training results
        self.train['workers'] = 8  # number of workers to load images
        self.train['gpu'] = [0]  # select gpu devices

        self.train['alpha'] = 0.0  # weight for variance term
        self.train['optimizer'] = 'adam'  # define optimizer
        self.train['validation'] = 0  # doing validation

        self.train['checkpoint_freq'] = 100  # epoch to save checkpoints
        # --- resume training --- #
        self.train['start_epoch'] = 0  # start epoch
        self.train['checkpoint'] = ''  # checkpoint to resume training or evaluation
        # 'random_resize' 'random_color' 'horizontal_flip' 'random_affine' 'random_elastic'
        # 'random_rotation' 'random_crop' 'label_encoding' 'to_tensor' 'normalize'
        self.train['trans_train'] = ['random_color',
                                     'random_chooseAug',
                                     'horizontal_flip',
                                     #'random_rotation',
                                     'random_elastic',
                                     'random_crop',
                                     'label_encoding',
                                     'to_tensor',
                                     #'normalize'
                                     ]

        self.transform_string = str(self.train['trans_train'])
        self.transform_str = get_transformString(self.train['trans_train'])

        # ============================================= stringFirst =================================================
        stringFirst = '0_' + self.model['modelName'] + '[' + self.model['backbone'] + ']'
        stringFirst = stringFirst + '[' + str(self.train['optimizer']) + ']'  # adam radam radam4s adamw
        stringFirst = stringFirst + '_sche' + '[' + str(self.train['scheduler']) + ']'
        stringFirst = stringFirst + '_3c' if self.model['multi_class'] == True else stringFirst + '_2c'
        # ============================================= inputInfo =================================================
        inputInfo = '_' + 'input' + str(self.train['input_size']) + 'over' + str(self.train['val_overlap']) + \
                    'bs' + str(self.train['batch_size']) + '_e' + str(self.train['num_epochs'])
        # ============================================= stringLast =================================================
        stringLast = ''
        stringLast = stringLast + '_seed' + str(self.train['seed'])
        stringLast = stringLast + 'ms'
        stringLast = stringLast + '_addWM' if self.model['add_weightMap'] == 1 else stringLast + '_noWM'
        stringLast = stringLast + '_br' + str(self.train['branch'])
        stringLast = stringLast + '_Dice' + str(self.model['dice'])
        stringLast = stringLast + '_b' + str(self.model['boundary_loss'])
        stringLast = stringLast + '_estop' + str(self.train['early_stop']) if self.train['early_stop'] > 0 else stringLast + '_noestop'
        stringLast = stringLast + '_isVal' if self.train['validation'] == 1 else stringLast + '_noVal'
        #stringLast = stringLast + self.transform_str # The local name is commented out first because it is too long
        stringLast = stringLast + '_a0' if self.train['alpha'] == 0 else stringLast + '_a' + str(int(self.train['alpha']))

        self.model['exp_filename'] = stringFirst + inputInfo# + stringLast #'exp_filename'

        self.train['data_dir'] = './data/{:s}'.format(self.dataset)  # path to data
        self.train['save_dir'] = './experiments/{:s}/{:s}'.format(self.dataset, self.model['exp_filename'])
        self.train['weight_map_dir'] = '{:s}/weight_maps'.format(self.train['data_dir'])
        #self.train['labels_WM_dir'] = '{:s}/labels_WM'.format(self.train['data_dir'])
        #self.train['labels_WM_boundary_dir'] = '{:s}/labels_WM5'.format(self.train['data_dir'])


        # --- data transform --- #
        self.transform = dict()
        # defined in parse function

        # --- post processing --- #
        self.post = dict()
        self.post['postproc'] = 0
        self.post['min_area'] =  20  # minimum area for an object
        self.post['radius'] = 2 # 3

        # --- test parameters --- #
        self.test = dict()
        self.test['filename'] = 'test1'  # test #test2
        self.test['epoch'] = 'best'
        self.test['gpu'] = [0]
        self.test['branch'] = 5

        self.test['groundtruth'] = 0
        self.test['img_dir'] = './data/{:s}/images/{:s}'.format(self.dataset,
                                                                self.test['filename'])
        self.test['label_dir'] = './data/{:s}/labels/{:s}'.format(self.dataset, self.test[
            'filename'])  # labels_instance
        self.test['annotation_dir'] = './data/{:s}/Annotations'.format(self.dataset)
        self.test['weight_map_dir'] = './data/{:s}/weight_maps'.format(self.dataset)

        self.test['tta'] = True  # True #False
        self.test['save_flag'] = True  # True #False
        self.test['patch_size'] = 256
        self.test['overlap'] = 40
        self.test['savefilename'] = 'br' + str(self.test['branch']) + '_' + self.test['filename'] + '_gt' + \
                                     str(self.test['groundtruth']) + '_post' + str(self.post['postproc']) + '_' + self.test['epoch'] + \
                                    '_minarea' + str(self.post['min_area']) + '_ra' + str(self.post['radius'])
        if (self.test['tta'] != True):
            self.test['savefilename'] = self.test['savefilename'] + '_notta'
        self.test['save_dir'] = './experiments/{:s}/{:s}/{:s}'.format(self.dataset, self.model['exp_filename'],
                                                                      self.test['savefilename'])
        self.test['model_path'] = './experiments/{:s}/{:s}/checkpoints/checkpoint_{:s}.pth.tar'.format(self.dataset,
                                                                                                       self.model[
                                                                                                           'exp_filename'],
                                                                                                       self.test[
                                                                                                           'epoch'])
        self.test['model_branch_path'] = './experiments/{:s}/{:s}/checkpoints/checkpointBranch_{:s}.pth.tar'.format(
            self.dataset,
            self.model[
                'exp_filename'],
            self.test[
                'epoch'])

        self.test['model_branch_path2'] = './experiments/{:s}/{:s}/checkpoints/checkpointBranch2_{:s}.pth.tar'.format(
            self.dataset,
            self.model[
                'exp_filename'],
            self.test[
                'epoch'])

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        if self.isTrain:
            parser.add_argument('--dataset', type=str, default=self.dataset,
                                help='input data set')
            parser.add_argument('--model-name', type=str, default=self.model['modelName'],
                                help='input model Name for training')
            parser.add_argument('--gpu', type=list, default=self.train['gpu'], help='GPUs for training')
            parser.add_argument('--branch', type=int, default=self.train['branch'], help='branch type')
            parser.add_argument('--epochs', type=int, default=self.train['num_epochs'],
                                help='number of epochs to train')
            parser.add_argument('--input-size', type=int, default=self.train['input_size'],
                                help='input size for training')
            parser.add_argument('--val-overlap', type=int, default=self.train['val_overlap'],
                                help='validation overlap size for training')
            parser.add_argument('--batch-size', type=int, default=self.train['batch_size'],
                                help='input batch size for training')
            parser.add_argument('--weight-map', type=int, default=self.model['add_weightMap'],
                                help='if add the weight map')
            
            parser.add_argument('--backbone', type=str, default=self.model['backbone'], help='model backbone')
            parser.add_argument('--pretrained', type=int, default=self.model['pretrained'], help='Whether to use pretrained model')
            parser.add_argument('--LossName', type=str, default=self.model['LossName'], help='all Loss name')
            parser.add_argument('--seed', type=int, default=self.train['seed'], help='random seed')
            parser.add_argument('--early_stop', type=int, default=self.train['early_stop'], help='let train early stop')
            parser.add_argument('--scheduler', type=str, default=self.train['scheduler'], help='name of the learning rate scheduler')
            parser.add_argument('--step', type=int, default=5, help='frequency of updating learning rate, given in epochs')
            parser.add_argument('--lr', type=float, default=self.train['lr'], help='learning rate')
            parser.add_argument('--lr_decay', type=float, default=self.train['lr_decay'], help='learning rate decay (default: 0.995)')
            parser.add_argument('--momentum', default=0.95, type=float, help='momentum')
            
            parser.add_argument('--optimizer', type=str, default=self.train['optimizer'],
                                help='input size for training')
            parser.add_argument('--alpha', type=float, default=self.train['alpha'],
                                help='The weight for the variance term in loss')
            parser.add_argument('--dice', type=int, default=self.model['dice'],
                                help='The other loss(e.g. dice loss)')
            parser.add_argument('--boundary-loss', type=int, default=self.model['boundary_loss'],
                                help='The other loss(e.g. boundary loss)')
            parser.add_argument('--mseloss', type=int, default=self.model['mseloss'],
                                help='The other loss(e.g. mse loss)')
            parser.add_argument('--direction', type=int, default=self.model['direction'],
                                help='The direction supervision')

            parser.add_argument('--log-interval', type=int, default=self.train['log_interval'],
                                help='how many batches to wait before logging training status')
            parser.add_argument('--data-dir', type=str, default=self.train['data_dir'],
                                help='directory of training data')
            parser.add_argument('--save-dir', type=str, default=self.train['save_dir'],
                                help='directory to save training results')  # (Replace with exp_filename)
            parser.add_argument('--checkpoint-path', type=str, default=self.train['checkpoint'],
                                help='directory to load a checkpoint')
            parser.add_argument('--transform-train', type=str, default=self.transform_str,
                                help='control transform')
            parser.add_argument('--exp-filename', type=str, default=self.model['exp_filename'],
                                help='exp_filename')
            parser.add_argument('--validation', type=int, default=self.train['validation'],
                                help='input size for training')
            
            parser.add_argument('--all_img_test', type=int, default=1, help='all_img_test')
            
            
            
            
            args = parser.parse_args()
            self.dataset = args.dataset
            self.model['modelName'] = args.model_name
            self.train['num_epochs'] = args.epochs
            self.train['input_size'] = args.input_size
            self.train['val_overlap'] = args.val_overlap
            self.train['batch_size'] = args.batch_size
            self.model['add_weightMap'] = args.weight_map
            
            self.model['backbone'] = args.backbone
            self.model['pretrained'] = args.pretrained
            self.model['LossName'] = args.LossName
            self.train['seed'] = args.seed
            self.train['early_stop'] = args.early_stop
            self.train['scheduler'] = args.scheduler
            self.train['step'] = args.step
            self.train['lr'] = args.lr
            self.train['lr_decay'] = args.lr_decay
            self.momentum = args.momentum
            
            self.train['alpha'] = args.alpha
            self.model['dice'] = args.dice
            self.model['boundary_loss'] = args.boundary_loss
            self.model['mseloss'] = args.mseloss
            self.model['direction'] = args.direction
            self.train['lr'] = args.lr
            self.train['log_interval'] = args.log_interval
            self.train['gpu'] = list(args.gpu)
            self.train['branch'] = args.branch
            self.train['checkpoint'] = args.checkpoint_path
            self.train['data_dir'] = './data/{:s}'.format(self.dataset)  # args.data_dir
            self.train['img_dir'] = '{:s}/images'.format(self.train['data_dir'])
            self.train['label_dir'] = '{:s}/labels'.format(self.train['data_dir'])
            self.train['weight_map_dir'] = '{:s}/weight_maps'.format(self.train['data_dir'])
            #self.train['labels_WM_dir'] = '{:s}/labels_WM'.format(self.train['data_dir'])
            #self.train['labels_WM_boundary_dir'] = '{:s}/labels_WM5'.format(self.train['data_dir'])

            self.train['trans_train'] = args.transform_train
            self.transform_string = str(self.train['trans_train'])
            self.transform_str = self.train['trans_train']
            
            self.train['validation'] = int(args.validation)
            self.train['optimizer'] = str(args.optimizer)
            self.model['exp_filename'] = args.exp_filename
            self.model['exp_filename'] = self.model['exp_filename'] + '_MSE' if self.model['mseloss'] == 1 else self.model['exp_filename']
            self.model['exp_filename'] = self.model['exp_filename'] + '_addDir' if self.model['direction'] == 1 else self.model['exp_filename']
            self.model['mean_std'] = 'mean_std'
            
            self.all_img_test = args.all_img_test
            print('======>>>{}, {}'.format(self.model['multi_class'], self.model['exp_filename'],))
            if('_3c' in self.model['exp_filename']):
                self.model['multi_class'] = True
            else:
                self.model['multi_class'] = False
            self.model['in_c'] = 1  if self.dataset == 'BBBC039V1' else 3
            self.model['out_c'] = 3 if self.model['multi_class'] == True else 1   # output channel

            self.train['save_dir'] = './experiments/{:s}/{:s}'.format(self.dataset, self.model['exp_filename'])
        
            if not os.path.exists(self.train['save_dir']):
                os.makedirs(self.train['save_dir'], exist_ok=True)

            # define data transforms for training
            self.transform['train'] = OrderedDict()
            self.transform['val'] = OrderedDict()

            if '_isRRe' in self.train['trans_train']:
                self.transform['train']['random_resize'] = [1, 2]
            if '_isRCo' in self.train['trans_train']:
                self.transform['train']['random_color'] = 1
            if '_isRA' in self.train['trans_train']:
                self.transform['train']['random_affine'] = 0.3
            if '_isHF' in self.train['trans_train']:
                self.transform['train']['horizontal_flip'] = True
            self.transform['train']['vertical_flip'] = True
            if '_isRE' in self.train['trans_train']:
                self.transform['train']['random_elastic'] = [6, 15]
            if '_isRRo' in self.train['trans_train']:
                self.transform['train']['random_rotation'] = 90
            if '_isCAu' in self.train['trans_train']:
                self.transform['train']['random_chooseAug'] = 1
            if '_isRCr' in self.train['trans_train']:
                self.transform['train']['random_crop'] = self.train['input_size']
            if '_isLE' in self.train['trans_train']:

                self.transform['train']['label_encoding'] = [self.model['out_c'], self.post['radius'], self.model['direction']]


            self.transform['train']['to_tensor'] = 1
            if '_isNorm' in self.train['trans_train']:
                self.transform['train']['normalize'] = np.load('{:s}/{:s}.npy'.format(self.train['data_dir'], self.model['mean_std']))

            self.transform['train']['label_encoding'] = [self.model['out_c'], self.post['radius'], self.model['direction']]
            self.transform['val'] = {'label_encoding': [self.model['out_c'], self.post['radius'], self.model['direction']], 'to_tensor': 1,}
            if '_isNorm' in self.train['trans_train']:
                self.transform['val']['normalize'] = np.load('{:s}/{:s}.npy'.format(self.train['data_dir'], self.model['mean_std']))

        else:
            parser.add_argument('--dataset', type=str, default=self.dataset,
                                help='input data set')
            parser.add_argument('--model-name', type=str, default=self.model['modelName'],
                                help='input model Name for training')
            parser.add_argument('--patch-size', type=int, default=self.test['patch_size'],
                                help='patch size for testing')
            parser.add_argument('--test-overlap', type=int, default=self.test['overlap'],
                                help='test overlap size for testing')
            parser.add_argument('--epoch', type=str, default=self.test['epoch'],
                                help='select the model used for testing')
            parser.add_argument('--save-flag', type=bool, default=self.test['save_flag'],
                                help='flag to save the network outputs and predictions')
            parser.add_argument('--gpu', type=list, default=self.test['gpu'], help='GPUs for training')
            parser.add_argument('--branch', type=int, default=self.test['branch'], help='branch type')

            parser.add_argument('--mseloss', type=int, default=self.model['mseloss'],
                                help='The other loss(e.g. mse loss)')
            parser.add_argument('--direction', type=int, default=self.model['direction'],
                                help='The direction supervision')

            parser.add_argument('--img-dir', type=str, default=self.test['img_dir'], help='directory of test images')
            parser.add_argument('--label-dir', type=str, default=self.test['label_dir'], help='directory of labels')
            parser.add_argument('--save-dir', type=str, default=self.test['save_dir'],
                                help='directory to save test results')
            parser.add_argument('--model-path', type=str, default=self.test['model_path'],
                                help='train model to be evaluated')
            parser.add_argument('--LossName', type=str, default=self.model['LossName'], help='all Loss name')
            parser.add_argument('--test-filename', type=str, default=self.test['filename'],
                                help='test filename')
            parser.add_argument('--exp-filename', type=str, default=self.model['exp_filename'],
                                help='exp_filename')
            parser.add_argument('--postproc', type=int, default=self.post['postproc'],
                                help='post process')
            parser.add_argument('--min-area', type=int, default=self.post['min_area'],
                                help='min area')
            parser.add_argument('--radius', type=int, default=self.post['radius'],
                                help='radius')
            parser.add_argument('--groundtruth', type=int, default=self.test['groundtruth'],
                                help='radius')
            parser.add_argument('--all_img_test', type=int, default=1, help='all_img_test')
            args = parser.parse_args()
            self.dataset = args.dataset
            self.model['modelName'] = args.model_name
            self.test['patch_size'] = args.patch_size
            self.test['overlap'] = args.test_overlap
            self.test['epoch'] = args.epoch
            self.test['gpu'] = list(args.gpu)
            self.test['branch'] = args.branch

            self.model['mseloss'] = args.mseloss
            self.model['direction'] = args.direction

            self.test['save_flag'] = args.save_flag
            self.model['LossName'] = args.LossName

            self.test['filename'] = args.test_filename
            self.model['exp_filename'] = args.exp_filename
            self.model['exp_filename'] = self.model['exp_filename'] + '_MSE' if self.model['mseloss'] == 1 else self.model['exp_filename']
            self.model['exp_filename'] = self.model['exp_filename'] + '_addDir' if self.model['direction'] == 1 else self.model['exp_filename']
            self.post['postproc'] = args.postproc
            self.post['min_area'] = args.min_area
            self.post['radius'] = args.radius
            self.test['groundtruth'] = int(args.groundtruth)
            self.all_img_test = args.all_img_test
            if('_3c' in self.model['exp_filename']):
                self.model['multi_class'] = True
            else:
                self.model['multi_class'] = False
            self.model['in_c'] = 1 if self.dataset == 'BBBC039V1' else 3
            self.model['out_c'] = 3 if self.model['multi_class'] == True else 1   # output channel
            
            self.save_testfilename = 'br' + str(self.test['branch']) + '_' + self.test['filename'] + '_gt' + str(self.test['groundtruth']) + \
                                    '_AIT' + str(self.all_img_test) + '_post' + str(self.post['postproc']) + \
                                     '_' + self.test['epoch'] + '_minarea' + str(self.post['min_area']) + '_ra' + str(self.post['radius'])

            self.test['img_dir'] = './data/{:s}/images/{:s}'.format(self.dataset, self.test['filename'])

            self.test['label_dir'] = './data/{:s}/labels/{:s}'.format(self.dataset, self.test['filename'])
            self.test['annotation_dir'] = './data/{:s}/Annotations'.format(self.dataset)  # Annotations


            self.test['save_dir'] = './experiments/{:s}/{:s}/{:s}'.format(self.dataset, self.model['exp_filename'],
                                                                          self.save_testfilename)


            self.test['model_path'] = './experiments/{:s}/{:s}/checkpoints/checkpoint_{:s}.pth.tar'.format(self.dataset,
                                                self.model['exp_filename'], self.test['epoch'])

            self.test['model_branch_path'] = './experiments/{:s}/{:s}/checkpoints/checkpointBranch_{:s}.pth.tar'.format(
                                                self.dataset, self.model['exp_filename'], self.test['epoch'])

            self.test['model_branch_path2'] = './experiments/{:s}/{:s}/checkpoints/checkpointBranch2_{:s}.pth.tar'.format(
                                                self.dataset, self.model['exp_filename'], self.test['epoch'])


            if not os.path.exists(self.test['save_dir']):
                os.makedirs(self.test['save_dir'], exist_ok=True)

            self.transform['test'] = OrderedDict()

            if('_noNorm' in self.test['save_dir']):
                self.transform['test'] = {
                    'to_tensor': 1,
                    #'normalize': np.load('{:s}/{:s}.npy'.format(self.train['data_dir'], self.model['mean_std']))
                }
            else:
                self.transform['test'] = {
                    'to_tensor': 1,
                    'normalize': np.load('{:s}/{:s}.npy'.format(self.train['data_dir'], self.model['mean_std']))
                }
            

    def print_options(self, logger=None):
        message = '\n'
        message += self._generate_message_from_options()
        if not logger:
            print(message)
        else:
            logger.info(message)

    def save_options(self):
        if self.isTrain:
            filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        else:
            filename = '{:s}/test_options.txt'.format(self.test['save_dir'])
        message = self._generate_message_from_options()
        file = open(filename, 'w')
        file.write(message)
        file.close()

    def _generate_message_from_options(self):
        message = ''
        message += '# {str:s} Options {str:s} #\n'.format(str='-' * 25)
        train_groups = ['model', 'train', 'transform']
        test_groups = ['model', 'test', 'post', 'transform']
        cur_group = train_groups if self.isTrain else test_groups

        for group, options in self.__dict__.items():
            if group not in train_groups + test_groups:
                message += '{:>20}: {:<35}\n'.format(group, str(options))
            elif group in cur_group:
                message += '\n{:s} {:s} {:s}\n'.format('*' * 15, group, '*' * 15)
                if group == 'transform':
                    for name, val in options.items():
                        if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                            message += '{:s}:\n'.format(name)
                            for t_name, t_val in val.items():
                                t_val = str(t_val).replace('\n', ',\n{:22}'.format(''))
                                message += '{:>20}: {:<35}\n'.format(t_name, str(t_val))
                else:
                    for name, val in options.items():
                        message += '{:>20}: {:<35}\n'.format(name, str(val))
        message += '# {str:s} End {str:s} #\n'.format(str='-' * 26)
        return message
