import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

input1 = 'FILE1.nii.gz' # first segmentation (ground truth)
input2   = 'FILE2.nii.gz' # second segmentation

reader = sitk.ImageFileReader()
reader.SetImageIO("NiftiImageIO")
reader.SetFileName(input1)
image_truth = reader.Execute()
image_truth_arr = sitk.GetArrayFromImage(image_truth)
print((image_truth_arr.shape))

# Assuming input NIfTI files have two masks, with two labels:
# 1 = kidney segmentation, 2 = tumour segmentation
image_truth_kidney = np.zeros(image_truth_arr.shape)
image_truth_kidney[image_truth_arr==1] = 1 

image_truth_tumour = np.zeros(image_truth_arr.shape)
image_truth_tumour[image_truth_arr==2] = 1 

#slice_id = 29
#curr_mask = image_truth_kidney[slice_id, :, :]
#imgplot = plt.imshow(curr_mask, cmap='gray')
#plt.show(block=True)

reader.SetFileName(input2)
image_seg = reader.Execute()
image_seg_arr = sitk.GetArrayFromImage(image_seg)

image_seg_kidney = np.zeros(image_seg_arr.shape)
image_seg_kidney[image_seg_arr==1] = 1
image_seg_kidney[image_seg_arr==3] = 1 

image_seg_tumour = np.zeros(image_seg_arr.shape)
image_seg_tumour[image_seg_arr==2] = 1 

#curr_mask = image_seg_kidney[slice_id, :, :]
#imgplot = plt.imshow(curr_mask, cmap='gray')
#plt.show(block=True)

mult_arr_kidney  = np.multiply(image_truth_kidney, image_seg_kidney)
mult_arr_tumour = np.multiply(image_truth_tumour, image_seg_tumour)

#curr_mask = mult_arr_kidney[slice_id, :, :]
#imgplot = plt.imshow(curr_mask, cmap='gray')
#plt.show(block=True)

dice_kidney = np.sum(mult_arr_kidney)*2.0 / (np.sum(image_seg_kidney) + np.sum(image_truth_kidney))
dice_tumour = np.sum(mult_arr_tumour)*2.0 / (np.sum(image_seg_tumour) + np.sum(image_truth_tumour))

print('Dice coeffficient for kidney %s ' % (dice_kidney))
print('Dice coeffficient for tumour %s ' % (dice_tumour))
    
