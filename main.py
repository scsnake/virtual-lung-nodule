
import numpy as np
import dicom
import glob
import os
import matplotlib.pyplot as plt
import SimpleITK as sitk


class CtVolume:
    def __init__(self):
        self.window = (-600, 1500)
        self.data = None
        self.gray = None
        self.origin = None
        self.spacing = None

    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''

    def load_itk(self, filename):
        # Reads the image using SimpleITK
        itkimage = sitk.ReadImage(filename)

        # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
        self.data = ct_scan = sitk.GetArrayFromImage(itkimage)
        self.gray = self.apply_window(self.data)
        # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
        self.origin = origin = np.array(list(reversed(itkimage.GetOrigin())))

        # Read the spacing along each dimension
        self.spacing = spacing = np.array(list(reversed(itkimage.GetSpacing())))

        return ct_scan, origin, spacing

    def load_dicom(self, dirname):

        dcm_files = glob.glob(os.path.join(dirname, '[!_]*.dcm'))
        if len(dcm_files)==0:
            dcm_files = glob.glob(os.path.join(dirname, '[!_]*'))
        assert len(dcm_files)>0
        f = dicom.read_file(dcm_files[0])

        s, i = f.RescaleSlope*1.0, f.RescaleIntercept
        # p = f.pixel_array

        data = np.zeros((len(dcm_files),f.Columns, f.Rows), np.int16)
        self.spacing = np.array([f.SliceThickness, f.PixelSpacing[1], f.PixelSpacing[0]])

        for ind, path in enumerate(dcm_files):
            f = dicom.read_file(path)

            p = f.pixel_array
            data[ind] = np.array(p)

        self.data = (data*s + i).astype(np.int16)

        # self.gray = self.apply_window(self.data)

    def pixel_to_absolute_coord(self, coord):
        '''
        :param coord: zero-based pixel coord x, y, z (slice number)
        :return:
        '''
        ret = []
        for i in range(3):
            ret[i] = self.origin[i] + self.spacing[i]*1.0*coord[i]
        return ret

    def absolute_to_pixel_coord(self, coord):
        ret = []
        for i in range(3):
            ret[i] = np.rint((coord[i]-self.origin[i])*1.0/self.spacing[i])

        return ret

    def apply_window(self, data):
        wl, ww = self.window
        data[data < wl - ww] = wl - ww
        data[data >= wl + ww] = wl + ww - 1
        return ((data - (wl - ww)) * 256.0 / (2 * ww)).astype(np.uint8)

    def crop(self, center_pixel_coord, shape, padding):
        ret = np.full(shape, padding)
        data_shape = self.data.shape

        r1 = np.zeros((3,2), np.int16) #source range
        r2 = np.zeros((3,2), np.int16) #destination range
        for i in range(3):
            if shape[i] % 2 ==0:
                len = shape[i]/2
                i1 = center_pixel_coord[i]-len+1
                i2 = center_pixel_coord[i]+len
            else:
                len = (shape[i]-1)/2
                i1 = center_pixel_coord[i] - len
                i2 = center_pixel_coord[i] + len
            
            if i1<0:
                r1[i,0]=0
                r2[i,0]=abs(i1)
            else:
                r1[i,0]=i1
                r2[i,0]=0
            
            if i2>data_shape[i]-1:
                r1[i,1]=data_shape[i]
                r2[i,1]=shape[i]-(i2-data_shape[i]+1)
            else:
                r1[i,1]=i2+1
                r2[i,1]=shape[i]
            
        ret[r2[0,0]:r2[0,1], r2[1,0]:r2[1,1], r2[2,0]:r2[2,1]]= \
            self.apply_window(self.data[r1[0,0]:r1[0,1], r1[1,0]:r1[1,1], r1[2,0]:r1[2,1]])

        return ret



dataset_folder = '' # dataset_folder/id/*.dcm
dataset_csv = '' # id,x,y,z in each line of csv
dataset_coord_type = 'absolute' # absolute or pixel
output_dim = (128, 128, 128)

if __name__=='__main__':
    dir = r'C:\20160320 T0161817768'
    d =  CtVolume()
    # d.load_dicom(dir)
    # d.data=np.load('ct.npy')
    d.load_itk('C:\\LSTK\\LKDS-00370_outputROI.mhd')
    # t = d.crop((200,200,200), (128, 128,128), 127)
    t = d.gray
    plt.axis('off')
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(t[int(t.shape[0]/9*i)], cmap='gray')

    plt.show()
    a=1
