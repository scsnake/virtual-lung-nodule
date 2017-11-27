# -*- coding: utf-8 -*-

import csv
import glob
import os
import random

import SimpleITK as sitk
import dicom
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu


class CtVolume:
    def __init__(self, data=None, origin=None, spacing=None):
        self.window = (-600, 1500)
        self.data = data
        self.gray = None
        self.origin = origin
        self.spacing = spacing
        self.nodules = []

    def load_nodule_info(self, filename):
        with open(filename, 'rb') as f:
            r = csv.reader(f, delimiter=',', quotechar='"')
            for row in r:
                self.nodules.append(row)

    def load_image_data(self, filename):
        if filename.endswith('.mhd'):
            try:
                self.data, self.origin, self.spacing = self.load_itk(filename)
            except:
                pass
        else:
            try:
                self.data, self.origin, self.spacing = self.load_dicom(filename)
            except:
                pass

    def nodule_in_VOI(self, volume):
        shape, origin, spacing = volume.data.shape, volume.origin, volume.spacing

        for nodule in self.nodules:
            is_outside = True
            for i in range(3):
                is_outside &= (origin[i] + spacing[i] * shape[i] < nodule.coord[i] - nodule.diameter or
                               nodule.coord[i] + nodule.diameter < origin[i])

            if not is_outside:
                return True

        return False

    def negative_background_sampler(self, shape, padding):
        while 100:
            center_pixel_coord = [random.randint(0, self.data.shape[i] - 1)
                                  for i in range(3)]
            crop = self.crop(center_pixel_coord, shape, padding)
            if not self.nodule_in_VOI(crop):
                return crop
        return None

    def nodule_blender(self, background, nodule):
        while 100:
            shape1, shape2 = background.data.shape, nodule.data.shape
            pad_nodule = np.zeros(shape1, background.data.dtype)

            cr = []
            for i in range(3):
                t = random.randint(0, shape1[i] - shape2[i] - 1)
                cr.append([t, t + shape2[i]])
            pad_nodule[cr[0][0]:cr[0][1]][cr[1][0]:cr[1][1]][cr[2][0]:cr[2][1]] = nodule.data

            bin_nodule = pad_nodule.astype(np.bool)
            bin_bg = background.data
            bin_bg[bin_bg < self.window[0]] = self.window[0]
            bin_bg[bin_bg >= self.window[1]] = self.window[1]
            bin_bg = threshold_otsu(bin_bg)

            overlap = np.logical_and(bin_bg, bin_nodule)
            overlap_ratio = np.sum(overlap) * 1.0 / np.sum(bin_nodule)

            if overlap_ratio > 0.5:
                continue

            ret = background
            ret.data = np.maximum.reduce([background.data, nodule.data])
            return ret


        return None

    def load_itk(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        ct_scan = sitk.GetArrayFromImage(itkimage)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ct_scan, origin, spacing

    def load_dicom(self, dirname):

        dcm_files = glob.glob(os.path.join(dirname, '[!_]*.dcm'))
        if len(dcm_files) == 0:
            dcm_files = glob.glob(os.path.join(dirname, '[!_]*'))
        assert len(dcm_files) > 0
        f = dicom.read_file(dcm_files[0])

        s, i = f.RescaleSlope * 1.0, f.RescaleIntercept
        # p = f.pixel_array

        data = np.zeros((len(dcm_files), f.Columns, f.Rows), np.int16)
        spacing = np.array([f.SliceThickness, f.PixelSpacing[1], f.PixelSpacing[0]])
        origin = reversed(f[0x20, 0x32].value)  # image position

        for ind, path in enumerate(dcm_files):
            f = dicom.read_file(path)

            p = f.pixel_array
            data[ind] = np.array(p)

        return (data * s + i).astype(np.int16), origin, spacing

        # self.gray = self.apply_window(self.data)

    def pixel_to_absolute_coord(self, coord):
        '''
        :param coord: zero-based pixel coord x, y, z (slice number)
        :return:
        '''
        ret = []
        for i in range(3):
            ret[i] = self.origin[i] + self.spacing[i] * 1.0 * coord[i]
        return ret

    def absolute_to_pixel_coord(self, coord):
        ret = []
        for i in range(3):
            ret[i] = np.rint((coord[i] - self.origin[i]) * 1.0 / self.spacing[i])

        return ret

    def apply_window(self, data):
        wl, ww = self.window
        data[data < wl - ww] = wl - ww
        data[data >= wl + ww] = wl + ww - 1
        return ((data - (wl - ww)) * 256.0 / (2 * ww)).astype(np.uint8)

    def crop(self, center_pixel_coord, shape, padding):

        ret = np.full(shape, padding)
        data_shape = self.data.shape

        r1 = np.zeros((3, 2), np.int16)  # source range
        r2 = np.zeros((3, 2), np.int16)  # destination range
        origin = []
        for i in range(3):
            if shape[i] % 2 == 0:
                len = shape[i] / 2
                i1 = center_pixel_coord[i] - len + 1
                i2 = center_pixel_coord[i] + len
            else:
                len = (shape[i] - 1) / 2
                i1 = center_pixel_coord[i] - len
                i2 = center_pixel_coord[i] + len

            if i1 < 0:
                r1[i, 0] = 0
                r2[i, 0] = abs(i1)
            else:
                r1[i, 0] = i1
                r2[i, 0] = 0

            if i2 > data_shape[i] - 1:
                r1[i, 1] = data_shape[i]
                r2[i, 1] = shape[i] - (i2 - data_shape[i] + 1)
            else:
                r1[i, 1] = i2 + 1
                r2[i, 1] = shape[i]

            origin[i] = self.origin[i] - (r1[i, 0] - r2[i, 0]) * self.spacing[i]

        ret[r2[0, 0]:r2[0, 1], r2[1, 0]:r2[1, 1], r2[2, 0]:r2[2, 1]] = \
            self.apply_window(self.data[r1[0, 0]:r1[0, 1], r1[1, 0]:r1[1, 1], r1[2, 0]:r1[2, 1]])

        return CtVolume(ret, origin, self.spacing)


dataset_folder = ''  # dataset_folder/id/*.dcm
dataset_csv = ''  # id,x,y,z in each line of csv
dataset_coord_type = 'absolute'  # absolute or pixel
output_dim = (128, 128, 128)

if __name__ == '__main__':
    dir = r'C:\20160320 T0161817768'
    d = CtVolume()
    d.load_dicom(dir)
    # d.data=np.load('ct.npy')
    # d.load_itk('C:\\LSTK\\LKDS-00370_outputROI.mhd')
    # t = d.crop((200,200,200), (128, 128,128), 127)
    # t = d.gray
    plt.axis('off')
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(t[int(t.shape[0] / 9 * i)], cmap='gray')

    plt.show()
    a = 1
