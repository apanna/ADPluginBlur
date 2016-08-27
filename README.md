ADPluginBlur
============

An EPICS areaDetector plugin to perform smoothing operation on NDArrays.

Additional information:

- Documentation.

- Input PVs

PV                 |  Comment
-------------------|---------
KernelWidth        | Width of the convolution kernel
KernelHeight       | Height of the convolution kernel
BlurType           | Type of smoothing filter. Currently supports Normalized box, Gaussian, Median and Bilateral

- Release notes.
  
  For this implementation the image must be 8 bit monochromatic.
  Fortunately there are some other plugins (ColorConvert and Process)
  that allow you to convert to this format.
  
  Requires opencv (http://opencv.org).
  This code was built using verion 3.0 of the opencv API.
