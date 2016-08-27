#ifndef NDPluginBlur_H
#define NDPluginBlur_H

#include "NDPluginDriver.h"

/** Map parameter enums to strings that will be used to set up EPICS databases */
#define NDPluginBlurKernelWidthString      "KERNEL_WIDTH"     /* (asynInt32, r/w)  width of the convolution kernel */
#define NDPluginBlurKernelHeightString     "KERNEL_HEIGHT"    /* (asynInt32, r/w)  height of the convolution kernel */
#define NDPluginBlurBlurTypeString         "BLUR_TYPE"        /* (asynInt32, r/w)  type of smoothing filter */

/** Does the blurring operations */
class NDPluginBlur : public NDPluginDriver {
public:
    NDPluginBlur(const char *portName, int queueSize, int blockingCallbacks, 
                 const char *NDArrayPort, int NDArrayAddr,
                 int maxBuffers, size_t maxMemory,
                 int priority, int stackSize);
    /* These methods override the virtual methods in the base class */
    void processCallbacks(NDArray *pArray);
    
protected:
    int NDPluginBlurKernelWidth;
    #define FIRST_NDPLUGIN_BLUR_PARAM NDPluginBlurKernelWidth
    int NDPluginBlurKernelHeight;
    int NDPluginBlurBlurType_;
    #define LAST_NDPLUGIN_BLUR_PARAM NDPluginBlurBlurType_

private:
    void blur(NDArray *inArray, NDArray *outArray, NDArrayInfo_t *arrayInfo);
};
#define NUM_NDPLUGIN_BLUR_PARAMS ((int)(&LAST_NDPLUGIN_BLUR_PARAM - &FIRST_NDPLUGIN_BLUR_PARAM + 1))
    
#endif
