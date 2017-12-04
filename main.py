# -*- coding: utf-8 -*-

import csv
import glob
import os
import random

import SimpleITK as sitk
import dicom
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom, generic_gradient_magnitude, sobel
from skimage.filters import threshold_otsu
from scipy.ndimage.filters import gaussian_filter

import pickle


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, rows, cols = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray')
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def ViewCT(arr):
    global ax, fig, tracker
    fig, ax = plt.subplots(1, 1)

    tracker = IndexTracker(ax, arr)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()


class CtVolume(object):
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
                if row[0] == self.id:
                    nodule = {}
                    nodule['coord'] = [float(row[3]), float(row[2]), float(row[2])]
                    nodule['diameter'] = float(row[4])
                    self.nodules.append(nodule)

    def load_image_data(self, filename):
        if filename.endswith('.mhd') or filename.endswith('.mha'):
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
                is_outside &= (origin[i] + spacing[i] * shape[i] < nodule['coord'][i] - nodule['diameter'] or
                               nodule['coord'][i] + nodule['diameter'] < origin[i])

            if not is_outside:
                return True

        return False

    def negative_background_sampler(self, shape, padding=None):
        lung_HU=(-900, -500)
        tried=0
        while 1:
            tried+=1
            if tried>1000:
                break
            if padding is None:
                dia=[int(shape[i]/2) for i in range(3)]
                center_pixel_coord = [random.randint(dia[i], self.data.shape[i] - 1-dia[i])
                                      for i in range(3)]
                padding=-1024
            else:
                center_pixel_coord = [random.randint(0, self.data.shape[i] - 1)
                                  for i in range(3)]
            crop = self.crop(center_pixel_coord, shape, padding)

            # if crop.data.min()>lung_HU[1] or crop.data.max()<lung_HU[0]:
            #     continue

            estimated_lung_parenchyma_ratio = ((lung_HU[0] < crop.data) & (crop.data < lung_HU[1])).sum() * 1.0 / (
                shape[0]*shape[1]*shape[2]
            )
            if estimated_lung_parenchyma_ratio<0.5:
                continue
            # ViewCT(crop.data)
            if self.nodule_in_VOI(crop):
                continue
            # print tried, center_pixel_coord
            return crop
        return None

    def nodule_blender(self, nodule, background=None):
        if background is None:
            background = self.negative_background_sampler((128, 128, 128), -1024)

        nodule_data = nodule.masked_data()
        shape1, shape2 = background.data.shape, nodule_data.shape

        tried = 0
        while 1:
            tried += 1
            if tried > 100:
                break
            pad_nodule = np.full(shape1, -1024, background.data.dtype)

            cr = []
            for i in range(3):
                t = random.randint(0, shape1[i] - shape2[i] - 1)
                cr.append([t, t + shape2[i]])
            pad_nodule[cr[0][0]:cr[0][1], cr[1][0]:cr[1][1], cr[2][0]:cr[2][1]] = nodule_data
            pad_nodule_edge = generic_gradient_magnitude(pad_nodule, sobel)

            bin_nodule = pad_nodule > -1024
            # pad_nodule_gradient_mask = pad_nodule_edge * 1.0 / pad_nodule_edge.max() + (bin_nodule*1.0)
            # pad_nodule_gradient_mask[pad_nodule_gradient_mask>1.0]=1.0

            non_overlap=pad_nodule>background.data

            # bin_bg = background.data
            # bin_bg[bin_bg < self.window[0]] = self.window[0]
            # bin_bg[bin_bg >= self.window[1]] = self.window[1]
            # th = threshold_otsu(bin_bg)
            # bin_bg = bin_bg > th
            #
            # overlap = np.logical_and(bin_bg, bin_nodule)
            non_overlap_ratio = np.sum(non_overlap) * 1.0 / np.sum(bin_nodule)

            if non_overlap_ratio < 0.5:
                continue

            ret = background
            # ret.data = background.data+pad_nodule
            ret.data = np.maximum.reduce([background.data, pad_nodule])
            # ret.data = background.data * (1.0 - pad_nodule_gradient_mask) + pad_nodule * pad_nodule_gradient_mask
            return ret

        return None

    def load_itk(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)
        self.id = os.path.splitext(os.path.basename(filename))[0]
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
        self.id = os.path.basename(dirname)
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

    def apply_window(self, data=None):
        if data is None:
            data = self.data
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

            origin.append(self.origin[i] - (r1[i, 0] - r2[i, 0]) * self.spacing[i])

        ret[r2[0, 0]:r2[0, 1], r2[1, 0]:r2[1, 1], r2[2, 0]:r2[2, 1]] = (self.data[r1[0, 0]:r1[0, 1], r1[1, 0]:r1[1, 1], r1[2, 0]:r1[2, 1]])

        return CtVolume(ret, origin, self.spacing)


class Nodule(CtVolume):
    def __init__(self, image_data, mask=None):
        super(Nodule, self).__init__()
        self.load_image_data(image_data)

        self.mask, origin, spacing = self.load_itk(mask) if mask else (None,None,None)
        self.mask = zoom(self.mask, (self.data.shape[0] * 1.0 / self.mask.shape[0], 1, 1))
        # print self.origin, self.spacing
        # print origin, spacing

    def masked_data(self, threshold=-3.2, fill=-1024):
        data = self.data
        data[self.mask < threshold] = fill
        return self.crop_image(data, fill)

    def crop_image(self, img, tol=0):
        mask = img > tol

        # Coordinates of non-black pixels.
        coords = np.argwhere(mask)

        # Bounding box of non-black pixels.
        x0, y0, z0 = coords.min(axis=0)
        x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top

        # Get the contents of the bounding box.
        return img[x0:x1, y0:y1, z0:z1]


dataset_folder = ''  # dataset_folder/id/*.dcm
dataset_csv = ''  # id,x,y,z in each line of csv
dataset_coord_type = 'absolute'  # absolute or pixel
output_dim = (128, 128, 128)

if __name__ == '__main__':
    dir = r'C:\20160320 T0161817768'
    d = CtVolume()
    # d.load_dicom(dir)
    # d.data=np.load('ct.npy')
    d.load_image_data(r'C:\LKDS\LKDS-00024.mhd')
    d.load_nodule_info(r'C:\LKDS\annotations_reviewed_sorted.csv')

    nd = Nodule(r'C:\LKDS\LKDS-00024_outputROI_ps.mhd',
                r'C:\LKDS\LKDS-00024_outputTumorImage_ps.mha')

    # d.load_itk(r'C:\LKDS\LKDS-00001.mhd')
    # t = d.crop((200,200,200), (128, 128,128), 127)
    # t = d.nodule_blender(nd)
    while 1:
        n = d.negative_background_sampler((64, 64, 64))
        t = d.nodule_blender(nd,n)
        t = nd.apply_window(t.data)
        # np.save('new.npy', t )

        ViewCT(t)
        a=3

    # plt.axis('off')
    # for i in range(16):
    #     plt.subplot(4, 4, i + 1)
    #     plt.imshow(t[int(t.shape[0] *1.0/ 16 * i)], cmap='gray')
    #
    # plt.show()

