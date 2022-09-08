import SimpleITK as sitk

import matplotlib.pyplot as plt
import numpy as np

import sys

if __name__ == '__main__':
    
  arguments = sys.argv[1:]
  if (len(arguments) < 1):
      print('Error. You need to provide an input file')
      exit()

  input_list = arguments[0]

  with open(input_list, 'r') as fp:
    line = (fp.readline()).strip()

    while line: 
      case_id = line.split('/')[-1]
      case_id = case_id.split('_')[-1]
      
      input_file = line + '/imaging.nii.gz'
      input_seg = line + '/segmentation.nii.gz'
      
      # read image
      reader = sitk.ImageFileReader()
      reader.SetImageIO("NiftiImageIO")
      reader.SetFileName(input_file)
      image = reader.Execute();
      image_arr = sitk.GetArrayFromImage(image)
      
      #OPT
      #curr_slice = image_arr[:, :, 196]
      #imgplot = plt.imshow(curr_slice, cmap='gray')
      #plt.show(block=True)
      
      # read segmentation
      reader2 = sitk.ImageFileReader()
      reader2.SetImageIO("NiftiImageIO")
      reader2.SetFileName(input_seg)
      segmentation = reader2.Execute();
      seg_arr = sitk.GetArrayFromImage(segmentation)
      
      #OPT
      #curr_seg = seg_arr[:, :, 196]
      #imgplot = plt.imshow(curr_seg, cmap='gray')
      #plt.show(block=True)

      output_name = 'kits19_' + case_id
      output_path = input_file.replace('imaging.nii.gz', output_name)
      np.savez_compressed(output_path, images=image_arr, segmentations=seg_arr)

      line = (fp.readline()).strip()
