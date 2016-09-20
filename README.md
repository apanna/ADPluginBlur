ADPluginBlur
============

An EPICS areaDetector plugin to perform smoothing operation on NDArrays.

Additional information:

- Documentation.
  (to-do)	

- Input PVs

PV                 |  Comment
-------------------|---------
KernelWidth        | Width of the convolution kernel
KernelHeight       | Height of the convolution kernel
BlurType           | Type of smoothing filter. Currently supports Normalized box, Gaussian, Median and Bilateral

- Release notes.
  
  Requires opencv (http://opencv.org).
  This code was built using verion 3.0 of the opencv API.
