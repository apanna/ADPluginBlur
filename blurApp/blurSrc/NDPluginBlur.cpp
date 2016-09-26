/*
 * NDPluginBlur.cpp
 *
 * Image blurring/smoothing plugin. Loosely based on ADPluginEdge.
 * Uses the opencv library.
 *
 * Author: Alireza Panna NIH/NHLBI/IPL
 *
 * Created Aug 24, 2016
 * TODO: Remove debug prints, thoroughly test the plugin.
 *
 * Change Log:
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#include <epicsString.h>
#include <epicsMutex.h>
#include <iocsh.h>

#include <asynDriver.h>

#include "NDArray.h"
#include "NDPluginDriver.h"
#include "NDPluginBlur.h"
#include <epicsExport.h>

#include <opencv2/opencv.hpp>

static const char *driverName = "NDPluginBlur";
/* Enums to describe the types of smoothing filter */
typedef enum 
{
  None,
  NormalizedBlock,
  Gaussian,
  Median,
  Bilateral,
} NDPluginBlurType_;

/** Perform the blur according to the smoothing filter */
void doBlur(NDArray *inArray, NDArray *outArray, NDArrayInfo_t *arrayInfo, int kernelWidth, int kernelHeight, int blurType)
{
  size_t numRows, rowSize;
  unsigned char *inData, *outData, *tempData;
  size_t nElements, elementSize;

  static const char* functionName = "doBlur";
  
  rowSize     = inArray->dims[arrayInfo->xDim].size;
  numRows     = inArray->dims[arrayInfo->yDim].size;
  nElements   = arrayInfo->nElements;
  elementSize = arrayInfo->bytesPerElement;
  // printf("In function: %s\n", functionName);
  // printf("rowSize=%d\nnumRows=%d\n", rowSize, numRows);
  // printf("Num Elements=%d\nbytes=%d\n", nElements, sizeof(*inData));
  // printf("Num Elements=%d\nelement size=%d\n", nElements, elementSize);
  // printf("bpp=%d\n", sizeof(&inArray));
  cv::Mat inImg = cv::Mat((int) numRows, (int) rowSize, CV_32F);
  cv::Mat outImg;
  /* inData has the original array */
  inData  = (unsigned char *)inArray->pData;
  /* temp data is formatted according to opencv::Mat */
  outData = (unsigned char *)outArray->pData;
  tempData = (unsigned char *)inImg.data;
  /* Copy the contents of the original array into inImg */
  memcpy(outData, inData, nElements * sizeof(float));
  memcpy(tempData, inData, nElements * sizeof(float));
  switch (blurType)
  {
    case (None):
      /* nothing to do here since the input array is already copied to the output array */
      break;

    case (NormalizedBlock):
      /* src image can be any of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F */
      try 
      {
        cv::blur(inImg, outImg, cv::Size(kernelWidth, kernelHeight), cv::Point(-1, -1));
        tempData = (unsigned char *)outImg.data;
        memcpy(outData, tempData, nElements * sizeof(float));
      }
      catch(cv::Exception &e) 
      {
        const char* err_msg = e.what();
        printf("%s::%s cv::blur exception: %s\n", driverName, functionName, err_msg);
      }
      break;

    case (Gaussian):
      /* src image can be any of CV_8U, CV_16U, CV_16S, CV_32F or CV_64F */
      try 
      {
        cv::GaussianBlur(inImg, outImg, cv::Size(kernelWidth, kernelHeight), 0, 0);
        tempData = (unsigned char *)outImg.data;
        memcpy(outData, tempData, nElements * sizeof(float));
      }
      catch( cv::Exception &e) 
      {
        const char* err_msg = e.what();
        printf("%s::%s cv::GaussianBlur exception: %s\n", driverName, functionName, err_msg);
      }
      break;

    case (Median):
      /* when ksize is 3 or 5, the image depth should be CV_8U, CV_16U, or CV_32F, for larger aperture sizes, it can only be CV_8U */
      try 
      {
        cv::medianBlur(inImg, outImg, kernelWidth);
        tempData = (unsigned char *)outImg.data;
        memcpy(outData, tempData, nElements * sizeof(float));
      }
      catch( cv::Exception &e) 
      {
        const char* err_msg = e.what();
        printf("%s::%s cv::medianBlur exception: %s\n", driverName, functionName, err_msg);
      }
      break;

    case (Bilateral):
      /* 8-bit or floating-point, 1-channel or 3-channel image. This needs to be tested */
      try 
      {
        cv::bilateralFilter(inImg, outImg, kernelWidth, 5, 5);
        tempData = (unsigned char *)outImg.data;
        memcpy(outData, tempData, nElements * sizeof(float));
      }
      catch( cv::Exception &e) 
      {
        const char* err_msg = e.what();
        printf("%s::%s cv::bilateralFilter exception: %s\n", driverName, functionName, err_msg);
      }
      break;

    default:
      break;
  }
  return;
}

/** Callback function that is called by the NDArray driver with new NDArray data.
  * Does image processing.
  * \param[in] pArray  The NDArray from the callback.
  */
void NDPluginBlur::processCallbacks(NDArray *pArray)
{
  NDArray *blurredArray;
  NDArrayInfo_t arrayInfo;
  int dataType;

  static const char* functionName = "processCallbacks";
  // printf("In function: %s\n", functionName);
  /* Call the base class method */
  NDPluginDriver::processCallbacks(pArray);
  /** Create a pointer to a structure of type NDArrayInfo_t and use it to get information about
      the input array.
  */
  pArray->getInfo(&arrayInfo);
  /* Previous version of the array was held in memory.  Release it and reserve a new one. */
  if (this->pArrays[0]) 
  {
    this->pArrays[0]->release();
    this->pArrays[0] = NULL;
  }
  /* Release the lock; this is computationally intensive and does not access any shared data */
  this->unlock();
  /* Copy the information from the current array */
  this->pArrays[0] = this->pNDArrayPool->copy(pArray, NULL, 1);
  blurredArray = this->pArrays[0];
  /* Convert to 32 bit float for smoothing operations, since all filters support float32 input arrays */
  this->pNDArrayPool->convert(pArray, &pArray, NDFloat32);
  this->pNDArrayPool->convert(pArray, &blurredArray, NDFloat32);
  // This plugin only works with 1-D or 2-D arrays
  switch (pArray->ndims) {
    case 1:
    case 2:
      this->blur(pArray, blurredArray, &arrayInfo);
      break;
    default:
      asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
        "%s::%s: error, number of array dimensions must be 1 or 2\n",
        driverName, functionName);
      return;
      break;
  }
  this->lock();
  getIntegerParam(NDDataType,     &dataType);
  /* Convert back to the input datatype */
  this->pNDArrayPool->convert(blurredArray, &blurredArray, (NDDataType_t)dataType);
  this->getAttributes(blurredArray->pAttributeList);
  doCallbacksGenericPointer(blurredArray, NDArrayData, 0);
  callParamCallbacks();
}


/** Blurs the image according to the selected choice.*/  
void NDPluginBlur::blur(NDArray *inArray, NDArray *outArray, NDArrayInfo_t *arrayInfo)
{
  int kernelWidth, kernelHeight, blurType;
  static const char* functionName = "blur";
  //printf("In function: %s\n", functionName);
  getIntegerParam(NDPluginBlurKernelWidth,   &kernelWidth);
  getIntegerParam(NDPluginBlurKernelHeight,  &kernelHeight);
  getIntegerParam(NDPluginBlurBlurType_,     &blurType);
  //printf("BLURTYPE=%d\nWIDTH=%d\nHEIGHT=%d\n", blurType, kernelWidth, kernelHeight);
  /* kernel width and height should be non negative and odd */ 
  if (kernelWidth <= 0 && kernelHeight <= 0)
  {
    asynPrint(pasynUserSelf, ASYN_TRACE_ERROR,
    "%s::%s: kernel width and height should be > 0\n",
    driverName, functionName);
    return;
  }
  if (kernelWidth % 2 == 0) 
  {
    kernelWidth += 1;
    setIntegerParam(NDPluginBlurKernelWidth,   kernelWidth);
  }
  if (kernelHeight % 2 == 0) 
  {
    kernelHeight += 1;
    setIntegerParam(NDPluginBlurKernelHeight,  kernelHeight);
  }
  // printf("datatype: %d\n", inArray->dataType);
  doBlur(inArray, outArray, arrayInfo, kernelWidth, kernelHeight, blurType);
  return;
}


/** Constructor for NDPluginBlur; most parameters are simply passed to NDPluginDriver::NDPluginDriver.
  * After calling the base class constructor this method sets reasonable default values for all of the
  * parameters.
  * \param[in] portName The name of the asyn port driver to be created.
  * \param[in] queueSize The number of NDArrays that the input queue for this plugin can hold when
  *            NDPluginDriverBlockingCallbacks=0.  Larger queues can decrease the number of dropped arrays,
  *            at the expense of more NDArray buffers being allocated from the underlying driver's NDArrayPool.
  * \param[in] blockingCallbacks Initial setting for the NDPluginDriverBlockingCallbacks flag.
  *            0=callbacks are queued and executed by the callback thread; 1 callbacks execute in the thread
  *            of the driver doing the callbacks.
  * \param[in] NDArrayPort Name of asyn port driver for initial source of NDArray callbacks.
  * \param[in] NDArrayAddr asyn port driver address for initial source of NDArray callbacks.
  * \param[in] maxBuffers The maximum number of NDArray buffers that the NDArrayPool for this driver is
  *            allowed to allocate. Set this to -1 to allow an unlimited number of buffers.
  * \param[in] maxMemory The maximum amount of memory that the NDArrayPool for this driver is
  *            allowed to allocate. Set this to -1 to allow an unlimited amount of memory.
  * \param[in] priority The thread priority for the asyn port driver thread if ASYN_CANBLOCK is set in asynFlags.
  * \param[in] stackSize The stack size for the asyn port driver thread if ASYN_CANBLOCK is set in asynFlags.
  */
NDPluginBlur::NDPluginBlur(const char *portName, int queueSize, int blockingCallbacks,
                         const char *NDArrayPort, int NDArrayAddr,
                         int maxBuffers, size_t maxMemory,
                         int priority, int stackSize)
    /* Invoke the base class constructor */
    : NDPluginDriver(portName, queueSize, blockingCallbacks,
                   NDArrayPort, NDArrayAddr, 1, NUM_NDPLUGIN_BLUR_PARAMS, maxBuffers, maxMemory,
                   asynInt32ArrayMask | asynFloat64ArrayMask | asynGenericPointerMask,
                   asynInt32ArrayMask | asynFloat64ArrayMask | asynGenericPointerMask,
                   ASYN_MULTIDEVICE, 1, priority, stackSize)
{   
  createParam( NDPluginBlurKernelWidthString,  asynParamInt32,  &NDPluginBlurKernelWidth);
  createParam( NDPluginBlurKernelHeightString, asynParamInt32,  &NDPluginBlurKernelHeight);
  createParam( NDPluginBlurBlurTypeString,     asynParamInt32,  &NDPluginBlurBlurType_);
  
  /* Set the plugin type string */
  setStringParam(NDPluginDriverPluginType, "NDPluginBlur");
  // Enable ArrayCallbacks.  
  setIntegerParam(NDArrayCallbacks, 1);
  /* Try to connect to the array port */
  connectToArrayPort();
}

/** Configuration command */
extern "C" int NDBlurConfigure(const char *portName, int queueSize, int blockingCallbacks,
                                 const char *NDArrayPort, int NDArrayAddr,
                                 int maxBuffers, size_t maxMemory,
                                 int priority, int stackSize)
{
    new NDPluginBlur(portName, queueSize, blockingCallbacks, NDArrayPort, NDArrayAddr,
                        maxBuffers, maxMemory, priority, stackSize);
    return(asynSuccess);
}

/* EPICS iocsh shell commands */
static const iocshArg initArg0 = { "portName",iocshArgString};
static const iocshArg initArg1 = { "frame queue size",iocshArgInt};
static const iocshArg initArg2 = { "blocking callbacks",iocshArgInt};
static const iocshArg initArg3 = { "NDArrayPort",iocshArgString};
static const iocshArg initArg4 = { "NDArrayAddr",iocshArgInt};
static const iocshArg initArg5 = { "maxBuffers",iocshArgInt};
static const iocshArg initArg6 = { "maxMemory",iocshArgInt};
static const iocshArg initArg7 = { "priority",iocshArgInt};
static const iocshArg initArg8 = { "stackSize",iocshArgInt};
static const iocshArg * const initArgs[] = {&initArg0,
                                            &initArg1,
                                            &initArg2,
                                            &initArg3,
                                            &initArg4,
                                            &initArg5,
                                            &initArg6,
                                            &initArg7,
                                            &initArg8};
static const iocshFuncDef initFuncDef = {"NDBlurConfigure",9,initArgs};
static void initCallFunc(const iocshArgBuf *args)
{
    NDBlurConfigure(args[0].sval, args[1].ival, args[2].ival,
                       args[3].sval, args[4].ival, args[5].ival,
                       args[6].ival, args[7].ival, args[8].ival);
}

extern "C" void NDBlurRegister(void)
{
    iocshRegister(&initFuncDef,initCallFunc);
}

extern "C" 
{
    epicsExportRegistrar(NDBlurRegister);
}
