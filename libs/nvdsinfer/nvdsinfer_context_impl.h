/**
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#ifndef __NVDSINFER_CONTEXT_IMPL_H__
#define __NVDSINFER_CONTEXT_IMPL_H__

#include <condition_variable>
#include <queue>
#include <memory>
#include <mutex>
#include <stdarg.h>

#include "cuda_runtime_api.h"
#include <NvInfer.h>
#include <NvCaffeParser.h>

#include <opencv2/objdetect/objdetect.hpp>

#include <nvdsinfer_context.h>
#include <nvdsinfer_custom_impl.h>
#include <nvdsinfer_utils.h>


/**
 * Implementation of the INvDsInferContext interface.
 */
class NvDsInferContextImpl : public INvDsInferContext
{
public:
    /**
     * Default constructor.
     */
    NvDsInferContextImpl();

    /**
     * Initializes the Infer engine, allocates layer buffers and other required
     * initialization steps.
     */
    NvDsInferStatus initialize(NvDsInferContextInitParams &initParams,
            void *userCtx, NvDsInferContextLoggingFunc logFunc);

private:
    /**
     * Free up resouces and deinitialize the inference engine.
     */
    ~NvDsInferContextImpl();

    /* Implementation of the public methods of INvDsInferContext interface. */
    NvDsInferStatus queueInputBatch(NvDsInferContextBatchInput &batchInput) override;
    NvDsInferStatus dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput) override;
    void releaseBatchOutput(NvDsInferContextBatchOutput &batchOutput) override;
    void fillLayersInfo(std::vector<NvDsInferLayerInfo> &layersInfo) override;
    void getNetworkInfo(NvDsInferNetworkInfo &networkInfo) override;
    const std::vector<std::vector<std::string>>& getLabels() override;
    void destroy() override;

    /* Other private methods. */
    NvDsInferStatus checkEngineParams(NvDsInferContextInitParams &initParams);
    NvDsInferStatus useEngineFile(NvDsInferContextInitParams &initParams);
    NvDsInferStatus generateTRTModel(NvDsInferContextInitParams &initParams,
            nvinfer1::IHostMemory *&gieModelStream);
    NvDsInferStatus readMeanImageFile(char *meanImageFilePath);
    NvDsInferStatus getBoundLayersInfo();
    NvDsInferStatus allocateBuffers();
    NvDsInferStatus parseLabelsFile(char *labelsFilePath);
    bool parseBoundingBox(
        std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
        NvDsInferNetworkInfo const &networkInfo,
        NvDsInferParseDetectionParams const &detectionParams,
        std::vector<NvDsInferObjectDetectionInfo> &objectList);
    bool parseAttributesFromSoftmaxLayers(
        std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
        NvDsInferNetworkInfo  const &networkInfo,
        float classifierThreshold,
        std::vector<NvDsInferAttribute> &attrList,
        std::string &attrString);
    void clusterAndFillDetectionOutputCV(NvDsInferDetectionOutput &output);
    void clusterAndFillDetectionOutputDBSCAN(NvDsInferDetectionOutput &output);
    NvDsInferStatus fillDetectionOutput(NvDsInferDetectionOutput &output);
    NvDsInferStatus fillClassificationOutput(NvDsInferClassificationOutput &output);
    NvDsInferStatus fillSegmentationOutput(NvDsInferSegmentationOutput &output);
    void releaseFrameOutput(NvDsInferFrameOutput &frameOutput);
    NvDsInferStatus initNonImageInputLayers();

    /* Input layer has a binding index of 0 */
    static const int INPUT_LAYER_INDEX = 0;

    /* Mutex to keep DLA IExecutionContext::enqueue theadsafe */
    static std::mutex DlaExecutionMutex;

    /** Unique identifier for the instance. This can be used to identify the
     * instance generating log and error messages. */
    unsigned int m_UniqueID;

    unsigned int m_MaxBatchSize;

    double m_NetworkScaleFactor;

    /** Input format for the network. */
    NvDsInferFormat m_NetworkInputFormat;

    NvDsInferNetworkType m_NetworkType;

    /* Network input information. */
    NvDsInferNetworkInfo m_NetworkInfo;

    bool m_UseDBScan;

    NvDsInferDBScanHandle m_DBScanHandle;

    /* Number of classes detected by the model. */
    unsigned int m_NumDetectedClasses;

    /* Detection / grouping parameters. */
    std::vector<NvDsInferDetectionParams> m_PerClassDetectionParams;
    NvDsInferParseDetectionParams m_DetectionParams;

    /* Vector for all parsed objects. */
    std::vector<NvDsInferObjectDetectionInfo> m_ObjectList;
    /* Vector of cv::Rect vectors for each class. */
    std::vector<std::vector<cv::Rect>> m_PerClassCvRectList;
    /* Vector of NvDsInferObjectDetectionInfo vectors for each class. */
    std::vector<std::vector<NvDsInferObjectDetectionInfo>> m_PerClassObjectList;

    float m_ClassifierThreshold;
    float m_SegmentationThreshold;

    /* Custom library implementation. */
    void *m_CustomLibHandle;
    NvDsInferParseCustomFunc m_CustomBBoxParseFunc;
    NvDsInferClassiferParseCustomFunc m_CustomClassifierParseFunc;
    nvinfer1::IPluginFactory *m_RuntimePluginFactory;

    unsigned int m_GpuID;
    bool m_DlaEnabled;

    /* Holds the string labels for classes. */
    std::vector<std::vector<std::string>> m_Labels;

    /* Logger for GIE info/warning/errors */
    class NvDsInferLogger : public nvinfer1::ILogger
    {
        void log(Severity severity, const char *msg) override ;
        public:
        NvDsInferContextImpl *handle;
    };
    NvDsInferLogger m_Logger;

    /* Custom unique_ptrs. These TensorRT objects will get deleted automatically
     * when the NvDsInferContext object is deleted. */
    nvinfer1::IRuntime *m_InferRuntime;
    nvinfer1::ICudaEngine *m_CudaEngine;
    nvinfer1::IExecutionContext *m_InferExecutionContext;

    cudaStream_t m_PreProcessStream;
    cudaStream_t m_InferStream;
    cudaStream_t m_BufferCopyStream;

    /* Vectors for holding information about bound layers. */
    std::vector<NvDsInferLayerInfo> m_AllLayerInfo;
    std::vector<NvDsInferLayerInfo> m_OutputLayerInfo;

    float *m_MeanDataBuffer;

    std::vector<void *> m_BindingBuffers;

    unsigned int m_OutputBufferPoolSize;

    /**
     * Holds information for one batch for processing.
     */
    typedef struct
    {
        std::vector<std::vector<uint8_t>> m_HostBuffers;
        std::vector<void *> m_DeviceBuffers;

        unsigned int m_BatchSize;
        cudaEvent_t m_CopyCompleteEvent = nullptr;
        bool m_BuffersWithContext = true;

        //NvDsInferContextReturnInputAsyncFunc m_ReturnFunc = nullptr;
        //void *m_ReturnFuncData = nullptr;
    } NvDsInferBatch;

    std::vector<NvDsInferBatch> m_Batches;

    /* Queues and synchronization members for processing multiple batches
     * in parallel.
     */
    std::mutex m_QueueMutex;
    std::condition_variable m_QueueCondition;
    std::queue<unsigned int> m_ProcessIndexQueue;
    std::queue<unsigned int> m_FreeIndexQueue;

    bool m_CopyInputToHostBuffers;

    /* Cuda Event for synchronizing input consumption by TensorRT CUDA engine. */
    cudaEvent_t m_InputConsumedEvent;
    /* Cuda Event for synchronizing completion of pre-processing. */
    cudaEvent_t m_PreProcessCompleteEvent;
    /* Cuda Event for synchronizing infer completion by TensorRT CUDA engine. */
    cudaEvent_t m_InferCompleteEvent;

    NvDsInferContextLoggingFunc m_LoggingFunc;

    void *m_UserCtx;

    bool m_Initialized;
};

/* Calls clients logging callback function. */
static inline void
callLogFunc(NvDsInferContextImpl *ctx, unsigned int uniqueID, NvDsInferLogLevel level,
        const char *func, NvDsInferContextLoggingFunc logFunc, void *logCtx,
        const char *fmt, ...)
{
    va_list args;
    va_start (args, fmt);
    char logMsgBuffer[_MAX_STR_LENGTH + 1];
    vsnprintf(logMsgBuffer, _MAX_STR_LENGTH, fmt, args);
    logFunc(ctx, uniqueID, level, func, logMsgBuffer, logCtx);
    va_end (args);
}

#define printMsg(level, tag_str, fmt, ...) \
    do { \
        char * baseName = strrchr((char *) __FILE__, '/'); \
        baseName = (baseName) ? (baseName + 1) : (char *) __FILE__; \
        if (m_LoggingFunc) \
        { \
            callLogFunc(this, m_UniqueID, level, __func__, m_LoggingFunc, \
                    m_UserCtx, fmt, ## __VA_ARGS__); \
        } \
        else \
        { \
            fprintf(stderr, \
                tag_str " NvDsInferContextImpl::%s() <%s:%d> [UID = %d]: " fmt "\n", \
                __func__, baseName, __LINE__, m_UniqueID, ## __VA_ARGS__); \
        } \
    } while (0)

#define printError(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_ERROR, "Error in", fmt, ##__VA_ARGS__); \
    } while (0)

#define printWarning(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_WARNING, "Warning from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printInfo(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_INFO, "Info from", fmt, ##__VA_ARGS__); \
    } while (0)

#define printDebug(fmt, ...) \
    do { \
        printMsg (NVDSINFER_LOG_DEBUG, "DEBUG", fmt, ##__VA_ARGS__); \
    } while (0)

#endif
