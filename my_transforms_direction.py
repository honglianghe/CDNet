""" Author: Hongliang He """

import torch
import random
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import numbers
import collections
from skimage import morphology
import SimpleITK as sitk
import time
import copy
from skimage import io
import albumentations as albu
import warnings
warnings.filterwarnings("ignore")
class Compose(object):
    """ Composes several transforms together. 一起组成几个变换
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms, selectorNameList):
        self.transforms = transforms
        self.selectorNameList = selectorNameList
    def __call__(self, imgs):
        number = 0
        for t in self.transforms:
            selectorName = str(self.selectorNameList[number])
            start_time = time.time()
            imgs = t(imgs)

            number = number + 1
        return imgs



class Scale(object):
    """Rescale the input PIL images to the given size. """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        pics = []
        for img in imgs:
            if isinstance(self.size, int):
                w, h = img.size
                if (w <= h and w == self.size) or (h <= w and h == self.size):
                    pics.append(img)
                    continue
                if w < h:
                    ow = self.size
                    oh = int(self.size * h / w)
                    pics.append(img.resize((ow, oh), self.interpolation))
                    continue
                else:
                    oh = self.size
                    ow = int(self.size * w / h)
                    pics.append(img.resize((ow, oh), self.interpolation))
            else:
                pics.append(img.resize(self.size, self.interpolation))
        return tuple(pics)



class RandomResize(object):
    """Randomly Resize the input PIL Image using a scale of lb~ub.
    Args:
        lb (float): lower bound of the scale
        ub (float): upper bound of the scale
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, lb=0.5, ub=1.5, interpolation=Image.BILINEAR):
        self.lb = lb
        self.ub = ub
        self.interpolation = interpolation

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL Images): Images to be scaled.
        Returns:
            PIL Images: Rescaled images.
        """

        for img in imgs:
            if not isinstance(img, Image.Image):
                raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        scale = random.uniform(self.lb, self.ub)
        # print scale

        w, h = imgs[0].size  # 第一维是宽，第二维是高
        ow = int(w * scale)
        oh = int(h * scale)


        do_albu = 1
        if(do_albu == 1):
            transf = albu.Resize(always_apply=False, p=1.0, height=oh, width=ow, interpolation=0)

            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2]) #np.expand_dims(imgs[2], axis=2)
            if (len(label.shape) == 2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            concat_map_transf = transf(image=np.array(concat_map))['image']

            image_transf = concat_map_transf[:, :, 0:3]
            weightmap_transf = concat_map_transf[:, :, 3]
            if (label.shape[2] == 1):
                #label = label.reshape(label.shape[0], label.shape[1], 1)
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics = []
            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            if scale < 1:
                padding_l = (w - ow)//2
                padding_t = (h - oh)//2
                padding_r = w - ow - padding_l
                padding_b = h - oh - padding_t
                padding = (padding_l, padding_t, padding_r, padding_b)

            pics = []
            for i in range(len(imgs)):
                img = imgs[i]
                img = img.resize((ow, oh), self.interpolation)
                if scale < 1:
                    img = ImageOps.expand(img, border=padding, fill=0)
                pics.append(img)



        return tuple(pics)



class RandomColor(object):
    def __init__(self, randomMin = 1, randomMax = 2):

        self.randomMin = randomMin
        self.randomMax = randomMax


    def __call__(self, imgs):
        out_imgs = list(imgs)
        img = imgs[0]
        #0.5
        random_factor = 1 + (np.random.rand()-0.5)

        color_image = ImageEnhance.Color(img).enhance(random_factor)
        random_factor = 1 + (np.random.rand()-0.5)

        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)
        random_factor = 1 + (np.random.rand()-0.5)

        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)
        random_factor = 1 + (np.random.rand()-0.5)

        img_output = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

        out_imgs[0] = img_output

        return tuple(out_imgs)



class RandomAffine(object):
    """ Transform the input PIL Image using a random affine transformation
        The parameters of an affine transformation [a, b, c=0
                                                    d, e, f=0]
        are generated randomly according to the bound, and there is no translation
        (c=f=0)
    Args:
        bound: the largest possible deviation of random parameters
    """

    def __init__(self, bound):
        if bound < 0 or bound > 0.5:
            raise ValueError("Bound is invalid, should be in range [0, 0.5)")

        self.bound = bound

    def __call__(self, imgs):
        img = imgs[0]
        x, y = img.size

        a = 1 + 2 * self.bound * (random.random() - 0.5)
        b = 2 * self.bound * (random.random() - 0.5)
        d = 2 * self.bound * (random.random() - 0.5)
        e = 1 + 2 * self.bound * (random.random() - 0.5)

        # correct the transformation center to image center
        c = -a * x / 2 - b * y / 2 + x / 2
        f = -d * x / 2 - e * y / 2 + y / 2

        trans_matrix = [a, b, c, d, e, f]

        pics = []
        for img in imgs:
            pics.append(img.transform((x, y), Image.AFFINE, trans_matrix))

        return tuple(pics)



class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """

        pics = []
        if random.random() < 0.5:
            for img in imgs:#imgs
                pics.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            return tuple(pics)
        else:
            return imgs


class RandomVerticalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        pics = []
        if random.random() < 0.5:
            for img in imgs:
                pics.append(img.transpose(Image.FLIP_TOP_BOTTOM))
            return tuple(pics)
        else:
            return imgs

class RandomElasticDeform(object):
    """ Elastic deformation of the input PIL Image using random displacement vectors
        drawm from a gaussian distribution
    Args:
        sigma: the largest possible deviation of random parameters
    """
    def __init__(self, num_pts=4, sigma=20):
        self.num_pts = num_pts
        self.sigma = sigma

    def __call__(self, imgs):
        pics = []

        do_albu = 1
        if (do_albu == 1):
            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
            if(len(label.shape)==2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            transf = albu.ElasticTransform(always_apply=False, p=1.0, alpha=1.0, sigma=50, alpha_affine=50,
                                           interpolation=0, border_mode=0,
                                           value=(0, 0, 0),
                                           mask_value=None, approximate=False)  # border_mode 用于指定插值算法

            concat_map_transf = transf(image=concat_map)['image']
            image_transf = concat_map_transf[:, :, :3]
            weightmap_transf = concat_map_transf[:, :, 3]
            if (label.shape[2] == 1):
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            img = np.array(imgs[0])
            if len(img.shape) == 3:
                img = img[:,:,0]

            sitkImage = sitk.GetImageFromArray(img, isVector=False)
            mesh_size = [self.num_pts]*sitkImage.GetDimension()
            tx = sitk.BSplineTransformInitializer(sitkImage, mesh_size)

            params = tx.GetParameters()
            paramsNp = np.asarray(params, dtype=float)
            paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * self.sigma

            paramsNp[0:int(len(params)/3)] = 0  # remove z deformations! The resolution in z is too bad

            params = tuple(paramsNp)
            tx.SetParameters(params)

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitkImage)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(tx)
            resampler.SetDefaultPixelValue(0)

            for img in imgs:
                is_expand = False
                if not isinstance(img, np.ndarray):
                    img = np.array(img)

                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=2)
                    is_expand = True

                img_deformed = np.zeros(img.shape, dtype=img.dtype)

                for i in range(img.shape[2]):
                    sitkImage = sitk.GetImageFromArray(img[:,:,i], isVector=False)
                    outimgsitk = resampler.Execute(sitkImage)
                    img_deformed[:,:,i] = sitk.GetArrayFromImage(outimgsitk)

                if is_expand:
                    img_deformed = img_deformed[:,:,0]
                # print img_deformed.dtype
                pics.append(Image.fromarray(img_deformed))


        return tuple(pics)

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=Image.BILINEAR, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, imgs):
        """
            imgs (PIL Image): Images to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        pics = []

        do_albu = 1
        if (do_albu == 1):
            image = np.array(imgs[0])
            weightmap = np.expand_dims(imgs[1], axis=2)
            label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
            if (len(label.shape) == 2):
                label = label.reshape(label.shape[0], label.shape[1], 1)
            concat_map = np.concatenate((image, weightmap, label), axis=2)

            transf = albu.Rotate(always_apply=False, p=1.0, limit=(-90, 90), interpolation=0, border_mode=0,
                             value=(0, 0, 0), mask_value=None)  # border_mode 用于指定插值算法

            concat_map_transf = transf(image=concat_map)['image']
            image_transf = concat_map_transf[:, :, :3]
            weightmap_transf = concat_map_transf[:, :, 3]
            if (label.shape[2] == 1):
                label_transf = concat_map_transf[:, :, -1:]
                label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
            else:
                label_transf = concat_map_transf[:, :, -3:]
            image_PIL = Image.fromarray(image_transf.astype(np.uint8))
            weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
            label_PIL = Image.fromarray(label_transf.astype(np.uint8))

            pics.append(image_PIL)
            pics.append(weightmap_PIL)
            pics.append(label_PIL)

        else:
            for img in imgs:
                pics.append(img.rotate(angle, self.resample, self.expand, self.center))

        return tuple(pics)




class RandomChooseAug(object):
    def __call__(self, imgs):

        pics = []

        p_value = random.random()


        if p_value < 0.25:  # 0.25
            pics.append(imgs[0].filter(ImageFilter.BLUR))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        elif p_value < 0.5:  # 0.5
            pics.append(imgs[0].filter(ImageFilter.GaussianBlur))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        elif p_value < 0.75:  # 0.75
            pics.append(imgs[0].filter(ImageFilter.MedianFilter))
            for k in range(1, len(imgs)):
                pics.append(imgs[k])
            return tuple(pics)

        else:
            return imgs



class RandomCrop(object):
    """Crop the given PIL.Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, size, padding=0, fill_val=(0,)):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.fill_val = fill_val

    def __call__(self, imgs):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        pics = []

        w, h = imgs[0].size
        th, tw = self.size
        if(th > h or tw > w):
            ow = tw
            oh = th

            do_albu = 1
            if(do_albu == 1):
                transf = albu.Resize(always_apply=False, p=1.0, height=oh, width=ow, interpolation=0)
                image = np.array(imgs[0])
                weightmap = np.expand_dims(imgs[1], axis=2)
                label = np.array(imgs[2]) #np.expand_dims(imgs[2], axis=2)
                if (len(label.shape) == 2):
                    label = label.reshape(label.shape[0], label.shape[1], 1)
                if(len(image.shape)==2):
                    image = image.reshape(image.shape[0], image.shape[1], 1)
                
                image_h, image_w = image.shape[:2]
                weightmap_h, weightmap_w = weightmap.shape[:2]
                label_h, label_w = label.shape[:2]

                if(image_h!=weightmap_h or image_h != label_h or image_w!=weightmap_w or image_w != label_w or weightmap_h!=label_h or weightmap_w!=label_w):

                    image_transf = np.resize(image,(th, tw, 3))
                    weightmap_transf = np.resize(weightmap,(th, tw))
                    label_transf = np.resize(label,(th, tw, 3))
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))
                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
                else:
                    concat_map = np.concatenate((image, weightmap, label), axis=2)

                    concat_map_transf = transf(image=np.array(concat_map))['image']
                    image_channel = image.shape[-1]
                    image_transf = concat_map_transf[:, :, :image_channel]
                    image_transf = np.squeeze(image_transf)
                    weightmap_transf = concat_map_transf[:, :, image_channel]
                    if (label.shape[2] == 1):
                        #label = label.reshape(label.shape[0], label.shape[1], 1)
                        label_transf = concat_map_transf[:, :, -1:]
                        label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
                    else:
                        label_transf = concat_map_transf[:, :, -3:]
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
        else:
            do_albu = 1
            if (do_albu == 1):
                min_max_height = (int(th * 0.6), th)
                transf = albu.RandomSizedCrop(always_apply=False, p=1.0, min_max_height=min_max_height, height=th, width=tw,
                                              w2h_ratio=1.0, interpolation=0)

                image = np.array(imgs[0])
                weightmap = np.expand_dims(imgs[1], axis=2)
                label = np.array(imgs[2])  # np.expand_dims(imgs[2], axis=2)
                if (len(label.shape) == 2):
                    label = label.reshape(label.shape[0], label.shape[1], 1)
                if(len(image.shape)==2):
                    image = image.reshape(image.shape[0], image.shape[1], 1)
                
                image_h, image_w = image.shape[:2]
                weightmap_h, weightmap_w = weightmap.shape[:2]
                label_h, label_w = label.shape[:2]

                if(image_h!=weightmap_h or image_h != label_h or image_w!=weightmap_w or image_w != label_w or weightmap_h!=label_h or weightmap_w!=label_w):

                    image_transf = np.resize(image,(th, tw, 3))
                    weightmap_transf = np.resize(weightmap,(th, tw))
                    label_transf = np.resize(label,(th, tw, 3))
                    
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)
                    
                else:
                
                    concat_map = np.concatenate((image, weightmap, label), axis=2)

                    concat_map_transf = transf(image=concat_map)['image']
                    image_channel = image.shape[-1]
                    image_transf = concat_map_transf[:, :, :image_channel]
                    image_transf = np.squeeze(image_transf)
                    weightmap_transf = concat_map_transf[:, :, image_channel]
                    if (label.shape[2] == 1):
                        label_transf = concat_map_transf[:, :, -1:]
                        label_transf = label_transf.reshape(label_transf.shape[0], label_transf.shape[1])
                    else:
                        label_transf = concat_map_transf[:, :, -3:]
                    image_PIL = Image.fromarray(image_transf.astype(np.uint8))
                    weightmap_PIL = Image.fromarray(weightmap_transf.astype(np.uint8))
                    label_PIL = Image.fromarray(label_transf.astype(np.uint8))

                    pics.append(image_PIL)
                    pics.append(weightmap_PIL)
                    pics.append(label_PIL)



            else:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)
                for k in range(len(imgs)):
                    img = imgs[k]
                    if self.padding > 0:
                        img = ImageOps.expand(img, border=self.padding, fill=self.fill_val[k])

                    if w == tw and h == th:
                        pics.append(img)
                        continue

                    pics.append(img.crop((x1, y1, x1 + tw, y1 + th)))


        return tuple(pics)




import skimage.filters.rank as sfr
from scipy.ndimage import filters, measurements
from skimage import morphology, io, color, measure, feature

from data_prepare.SegFix_offset_helper import DTOffsetHelper
from data_prepare.SegFix_offset_helper import Sobel
from scipy import ndimage
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, binary_dilation, binary_fill_holes
from skimage.morphology import remove_small_objects, watershed
import postproc_other

from data_prepare.getDirectionDiffMap import generate_dd_map
import cv2
from numba import jit

import math
@jit(nopython=True)
def get_centerpoint2(mask, n, m):
    #print(dis.shape)
    now = -1
    x = -1
    y = -1
    P = []
    for i in range(8):
        P.append((math.sin(2 * math.pi / 8 * i), math.cos(2 * math.pi / 8 * i)))
    for i in range(n):
        for j in range(m):
            if mask[i][j] > 0:
                
                ma = 0
                mi = 10000000
                for k in range(8):
                    l = 0
                    r = 1000
                    for tim in range(30):
                        mid = (l + r) / 2
                        nx = round(i + P[k][0] * mid)
                        ny = round(j + P[k][1] * mid)
                        if (nx >= 0 and nx < n and ny >= 0 and ny < m and mask[nx][ny] > 0):
                            l = mid
                        else:
                            r = mid
                    ma = max(ma, r)
                    mi = min(mi, r)
                assert(ma > 0 and mi > 0)
                centerness = mi / ma
                if centerness > now:
                    now = centerness
                    x = i
                    y = j
    return [int(x), int(y)]

class LabelEncoding(object):
    """
    Encoding the label, computes boundary individually
    """

    def __init__(self, out_c=3, radius=1, do_direction=0):
        self.out_c = out_c
        self.radius = 1#radius
        self.do_direction = do_direction

    def __call__(self, imgs):
        start_time = time.time()
        time_str = str(start_time)[-5:]

        out_imgs = list(imgs)
        label = imgs[2]  # imgs[-1]


        if not isinstance(label, np.ndarray):
            label = np.array(label)


        min_value = 190
        max_value = 210
        half_value = 255 * 0.5  # 0

        # if unique>2，input = instance level
        if(len(label.shape)==2):
            label_inside = label
            label_level_len = len(np.unique(label))
        else:
            label_inside = label[:, :, 0]
            label_level_len = len(np.unique(label_inside))

        if (self.out_c != 3):
            if (label_level_len > 2):
                ins3channel = measure.label(label)
                label_instance = ins3channel[:, :, 0]
                label_instance = measure.label(label_instance)
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label_instance > 0] = 2  # inside
                new_label_inside = copy.deepcopy(new_label)
                # boun_instance = morphology.dilation(label_instance) & (~morphology.erosion(label_instance, morphology.disk(self.radius)))
                # new_label[boun_instance > 0] = 2
            else:
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label[:, :, 0] > half_value] = 2  # inside
                new_label[label[:, :, 1] > half_value] = 2  # inside
                new_label = morphology.erosion(new_label, morphology.disk(self.radius))
                new_label_inside = copy.deepcopy(new_label)
                label_instance = measure.label(new_label_inside)
                # boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(self.radius)))
                # new_label[boun > 0] = 2  # boundary

        else:
            # if label_level_len>2，input = instance level
            if (label_level_len > 2):
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label_inside > 0] = 1  # inside
                new_label = remove_small_objects(new_label, 5)

                new_label_inside = copy.deepcopy(new_label)

                boun_instance = morphology.dilation(label_inside) & (~morphology.erosion(label_inside, morphology.disk(self.radius)))
                new_label[boun_instance > 0] = 2
                postproc = 1
                if (postproc == 0):
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = measure.label(label_inside_new)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))
                else:
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = postproc_other.process(label_inside_new.astype(np.uint8) * 255, model_mode='modelName', min_size=5)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))


            else:
                #print('输入的是3分类 label')
                new_label = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                new_label[label_inside > half_value] = 1  # inside
                new_label_inside = copy.deepcopy(new_label)
                boun = morphology.dilation(new_label) & (~morphology.erosion(new_label, morphology.disk(self.radius)))
                new_label[boun > 0] = 2  # boundary
                postproc = 0
                if (postproc == 0):
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = measure.label(label_inside_new)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))

                else:
                    label_inside_new = (new_label == 1).astype(np.uint8)
                    label_instance = postproc_other.process(label_inside_new.astype(np.uint8) * 255, model_mode='modelName', min_size=5)
                    label_instance = morphology.dilation(label_instance, selem=morphology.selem.disk(self.radius))

        label1 = Image.fromarray((new_label / 2 * 255).astype(np.uint8))
        out_imgs[2] = label1


        do_direction = self.do_direction
        if (do_direction == 1):
            height, width = label.shape[0], label.shape[1]
            distance_map = np.zeros((height, width), dtype=np.float)
            distance_center_map = np.zeros((height, width), dtype=np.float)

            dir_map = np.zeros((height, width, 2), dtype=np.float32)
            ksize = 11
            point_number = 0
            label_point = np.zeros((height, width), dtype=np.float)

            mask = label_instance
            markers_unique = np.unique(label_instance)
            markers_len = len(np.unique(label_instance)) - 1

            for k in markers_unique[1:]:
                nucleus = (mask == k).astype(np.int)
                distance_i = distance_transform_edt(nucleus)
                distance_i_normal = distance_i / distance_i.max()
                distance_map = distance_map + distance_i_normal

                #local_maxi = feature.peak_local_max(distance_i, exclude_border=0, num_peaks=1)
                #if (local_maxi.shape[0] != 1):
                #    print(local_maxi)
                #assert local_maxi.shape[0] > 0
                #assert nucleus[local_maxi[0][0], local_maxi[0][1]] > 0
                #label_point[local_maxi[0][0], local_maxi[0][1]] = 255.0
                
                center = get_centerpoint2(nucleus, nucleus.shape[0], nucleus.shape[1])
                local_maxi = [center]
                assert nucleus[center[0], center[1]] > 0
                label_point[center[0], center[1]] = 255.0
                
                #if (do_direction == 1):
                nucleus = morphology.dilation(nucleus, morphology.disk(self.radius))
                point_map_k = np.zeros((height, width), dtype=np.int)
                point_map_k[local_maxi[0][0], local_maxi[0][1]] = 1
                int_pos = distance_transform_edt(1 - point_map_k)
                int_pos = int_pos * nucleus
                distance_center_i = (1 - int_pos / (int_pos.max() + 0.0000001)) * nucleus
                distance_center_map = distance_center_map + distance_center_i

                dir_i = np.zeros_like(dir_map)
                sobel_kernel = Sobel.kernel(ksize=ksize)
                dir_i = torch.nn.functional.conv2d(
                    torch.from_numpy(distance_center_i).float().view(1, 1, height, width),
                    sobel_kernel, padding=ksize // 2).squeeze().permute(1, 2, 0).numpy()
                dir_i[(nucleus == 0), :] = 0
                dir_map[(nucleus != 0), :] = 0
                dir_map += dir_i
                point_number = point_number + 1
            assert int(label_point.sum() / 255) == markers_len
            
            t_time = time.time()
            
            distance_map = distance_center_map

            label_point_gaussian = ndimage.gaussian_filter(label_point, sigma=2, order=0).astype(np.float16)
            out_imgs.append(label_point_gaussian)
            


            # 角度
            angle = np.degrees(np.arctan2(dir_map[:, :, 0], dir_map[:, :, 1]))
            label_angle = copy.deepcopy(angle)
            label_angle[new_label_inside == 0] = -180
            label_angle = label_angle + 180
            angle[new_label_inside == 0] = 0
            vector = (DTOffsetHelper.angle_to_vector(angle, return_tensor=False))
            # direction class
            label_direction = DTOffsetHelper.vector_to_label(vector, return_tensor=False)
            label_direction_new = copy.deepcopy(label_direction)
            # input = instance level
            if (label_level_len > 2):
                label_direction_new[new_label_inside == 0] = -1
            # input = 3-class level

            else:
                #label_direction_new[new_label == 0] = -1
                label_direction_new[new_label_inside == 0] = -1
            label_direction_new2 = label_direction_new + 1

            direction_label = True  # True False
            if (direction_label == False):
                out_imgs.append(label_angle)
            else:
                out_imgs.append(label_direction_new2)




        else:
            min_value = 190
            # label_point_gaussian = np.zeros((height, width), dtype=np.float)
            # label_direction_new = np.zeros((height, width), dtype=np.float)
            # out_imgs.append(label_point_gaussian)
            # out_imgs.append(label_direction_new)



        return tuple(out_imgs)



class ToTensor(object):
    """ Convert (img, label) of type ``PIL.Image`` or ``numpy.ndarray`` to tensors.
    Converts img of type PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    Converts label of type PIL.Image or numpy.ndarray (H x W) in the range [0, 255]
    to a torch.LongTensor of shape (H x W) in the range [0, 255].
    """

    def __init__(self, index=1):
        self.index = index  # index to distinguish between images and labels

    def __call__(self, imgs):
        """
        Args:
            imgs (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if len(imgs) < self.index:
            raise ValueError('The number of images is smaller than separation index!')
        pics = []

        # process image
        for i in range(0, self.index):
            img = imgs[i]

            if (img.mode == 'RGBA'):
                img = img.convert('RGB')

            if isinstance(img, np.ndarray):
                # handle numpy array
                pic = torch.from_numpy(img.transpose((2, 0, 1)))
                # backward compatibility
                pics.append(pic.float().div(255))


            # handle PIL Image
            if img.mode == 'I':
                pic = torch.from_numpy(np.array(img, np.int32, copy=False))
            elif img.mode == 'I;16':
                pic = torch.from_numpy(np.array(img, np.int16, copy=False))
            else:
                pic = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if img.mode == 'YCbCr':  #
                nchannel = 3
            elif img.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(img.mode)

            pic = pic.view(img.size[1], img.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            pic = pic.transpose(0, 1).transpose(0, 2).contiguous()
            if isinstance(pic, torch.ByteTensor):
                pics.append(pic.float().div(255))
            else:
                pics.append(pic)

        # process labels:
        for i in range(self.index, len(imgs)):
            # process label
            label = imgs[i]
            if (img.mode == 'RGBA'):
                img = img.convert('RGB')
            if isinstance(label, np.ndarray):
                # handle numpy array
                label_tensor = torch.from_numpy(label)
                # backward compatibility
                pics.append(label_tensor)  # label_tensor.long()
                continue

            # handle PIL Image
            if label.mode == 'I':
                label_tensor = torch.from_numpy(np.array(label, np.int32, copy=False))
            elif label.mode == 'I;16':
                label_tensor = torch.from_numpy(np.array(label, np.int16, copy=False))
            else:
                label_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(label.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if label.mode == 'YCbCr':  #
                nchannel = 3
            elif label.mode == 'I;16':
                nchannel = 1
            else:
                nchannel = len(label.mode)
            label_tensor = label_tensor.view(label.size[1], label.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            label_tensor = label_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            # label_tensor = label_tensor.view(label.size[1], label.size[0])
            pics.append(label_tensor.long())

        return tuple(pics)




class Normalize(object):
    """ Normalize an tensor image with mean and standard deviation.
    Given mean and std, will normalize each channel of the torch.*Tensor,
     i.e. channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    ** only normalize the first image, keep the target image unchanged
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensors):
        """
        Args:
            tensors (Tensor): Tensor images of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensors = list(tensors)
        for t, m, s in zip(tensors[0], self.mean, self.std):
            t.sub_(m).div_(s)
        return tuple(tensors)











    
    
    





selector = {
    'scale': lambda x: Scale(x),
    'random_resize': lambda x: RandomResize(x[0], x[1]),
    'random_color': lambda x: RandomColor(x), #change to later
    'horizontal_flip': lambda x: RandomHorizontalFlip(),
    'vertical_flip': lambda x: RandomVerticalFlip(),
    'random_rotation': lambda x: RandomRotation(x),
    'random_elastic': lambda x: RandomElasticDeform(x[0], x[1]),
    'random_chooseAug' : lambda x: RandomChooseAug(),
    'random_crop': lambda x: RandomCrop(x),
    'random_affine': lambda x: RandomAffine(x),
    'label_encoding': lambda x: LabelEncoding(x[0], x[1], x[2]),
    'to_tensor': lambda x: ToTensor(x),
    'normalize': lambda x: Normalize(x[0], x[1])
}


def get_transforms(param_dict):
    """ data transforms for train, validation or test """
    start_time = time.time()

    t_list = []
    selectorNameList = []
    selectorName_str = ''
    for k, v in param_dict.items():
        t_list.append(selector[k](v))
        selectorNameList.append(k)
        selectorName_str = selectorName_str + '_' + str(k)

    returnValue = Compose(t_list, selectorNameList)
    # =======================================================================================================================
    end_time = time.time()
    work_time = end_time - start_time
    work_time = round(work_time, 2) / 60
    print('\t\t ======  Compose({:s}),  ====== each epoch work time is [{:.4f} min].'.format(selectorName_str, work_time))
    start_time = end_time
    # =======================================================================================================================


    return returnValue
