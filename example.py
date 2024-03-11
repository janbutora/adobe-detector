# -*- coding: UTF-8 -*-
"""
@author: Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2024
"""
import numpy as np
from glob import glob
import os
from bruteforce_adobe_pattern import *
from detectAdobe import *

def main():
  FPR = 1e-5
  w_path = './w.npy'

  if os.path.exists(w_path):
    w = np.load(w_path)
  else:
    print('Estimating the Adobe pattern...')
    w = estimate_pattern()
    np.save(w_path, w)

  im_list = np.array(sorted(glob("./test_images/*.jpg")))
  pattern_present = np.zeros((len(im_list)), dtype=bool)
  statistic = np.zeros((len(im_list)))
  for i, im_name in enumerate(im_list):
      pattern_present[i], statistic[i] = detect_adobe_pattern(im_name, w, FPR)

  print('Images:', im_list)
  print('Test statistics:', statistic)
  print('Decision threshold for FPR {} : {}'.format(FPR, get_threshold(FPR, 0, 1/np.sqrt(w.size))))
  print('Adobe pattern detected in:', im_list[pattern_present])

if __name__ == '__main__':
  main()
