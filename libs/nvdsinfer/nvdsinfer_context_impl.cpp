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

#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <dlfcn.h>
#include <unistd.h>

#include "nvtx3/nvToolsExtCudaRt.h"

#include "nvdsinfer_context_impl.h"
#include "nvdsinfer_conversion.h"

#include <NvInferPlugin.h>
#include <NvUffParser.h>
#include <NvOnnxParser.h>

/* Function types for custom library interfaces. */

using NvDsInferPluginFactoryCaffeGetFcn = decltype (&NvDsInferPluginFactoryCaffeGet);
using NvDsInferPluginFactoryCaffeDestroyFcn = decltype (&NvDsInferPluginFactoryCaffeDestroy);

using NvDsInferPluginFactoryUffGetFcn = decltype (&NvDsInferPluginFactoryUffGet);
using NvDsInferPluginFactoryUffDestroyFcn = decltype (&NvDsInferPluginFactoryUffDestroy);

using NvDsInferPluginFactoryRuntimeGetFcn = decltype (&NvDsInferPluginFactoryRuntimeGet);
using NvDsInferPluginFactoryRuntimeDestroyFcn = decltype (&NvDsInferPluginFactoryRuntimeDestroy);

using NvDsInferInitializeInputLayersFcn = decltype (&NvDsInferInitializeInputLayers);

using NvDsInferCudaEngineGetFcn = decltype (&NvDsInferCudaEngineGet);

/* Pair data type for returning input back to caller. */
using NvDsInferReturnInputPair = std::pair<NvDsInferContextReturnInputAsyncFunc, void *>;

static const int WORKSPACE_SIZE = 450 * 1024 * 1024;

using namespace nvinfer1;
using namespace std;

/*
 * TensorRT INT8 Calibration implementation. This implementation requires
 * pre-generated INT8 Calibration Tables. Please refer TensorRT documentation
 * for information on the calibration tables and the procedure for creating the
 * tables.
 *
 * Since this implementation only reads from pre-generated calibration tables,
 * readCalibrationCache is requires to be implemented.
 */
class NvDsInferInt8Calibrator : public IInt8EntropyCalibrator2
{
public:
    NvDsInferInt8Calibrator(string calibrationTableFile) :
            m_CalibrationTableFile(calibrationTableFile)
    {
    }

    ~NvDsInferInt8Calibrator()
    {
    }

    int
    getBatchSize() const override
    {
        return 0;
    }

    bool
    getBatch(void* bindings[], const char* names[], int nbBindings) override
    {
        return false;
    }

    /* Reads calibration table file contents into a buffer and returns a pointer
     * to the buffer.
     */
    const void*
    readCalibrationCache(size_t& length) override
    {
        m_CalibrationCache.clear();
        ifstream input(m_CalibrationTableFile, std::ios::binary);
        input >> noskipws;
        if (input.good())
            copy(std::istream_iterator<char>(input),
                      istream_iterator<char>(),
                      back_inserter(m_CalibrationCache));

        length = m_CalibrationCache.size();
        return length ? m_CalibrationCache.data() : nullptr;
    }

    void
    writeCalibrationCache(const void* cache, size_t length) override
    {
    }

private:
    string m_CalibrationTableFile;
    vector<char> m_CalibrationCache;
};

/**
 * Get the size of the element from the data type
 */
inline unsigned int
getElementSize(NvDsInferDataType t)
{
    switch (t)
    {
        case INT32:
            return 4;
        case FLOAT:
            return 4;
        case HALF:
            return 2;
        case INT8:
            return 1;
    }

    return 0;
}

static inline bool
string_empty(char *str)
{
    return strlen(str) == 0;
}

static inline bool
file_accessible (char *path)
{
    return (access(path, F_OK) != -1);
}

/* Cuda callback function for returning input back to client. */
static void
returnInputCudaCallback(cudaStream_t stream,  cudaError_t status, void*  userData)
{
    NvDsInferReturnInputPair *pair = (NvDsInferReturnInputPair  *) userData;
    pair->first(pair->second);
    delete pair;
}

std::mutex NvDsInferContextImpl::DlaExecutionMutex;

void
NvDsInferContextImpl::NvDsInferLogger::log(Severity severity, const char *msg)
{
    NvDsInferLogLevel level;

    switch (severity)
    {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            level = NVDSINFER_LOG_ERROR;
            break;
        case Severity::kWARNING:
            level = NVDSINFER_LOG_WARNING;
            break;
        case Severity::kINFO:
            level = NVDSINFER_LOG_DEBUG;
            break;
        default:
            return;
    }

    callLogFunc(handle, handle->m_UniqueID, level, __func__, handle->m_LoggingFunc,
            handle->m_UserCtx, msg);
}

/* Default constructor. */
NvDsInferContextImpl::NvDsInferContextImpl() :
        INvDsInferContext(),
        m_UniqueID(0),
        m_DBScanHandle(nullptr),
        m_CustomLibHandle(nullptr),
        m_CustomBBoxParseFunc(nullptr),
        m_CustomClassifierParseFunc(nullptr),
        m_RuntimePluginFactory(nullptr),
        m_GpuID (0),
        m_DlaEnabled (false),
        m_InferRuntime(nullptr),
        m_CudaEngine(nullptr),
        m_InferExecutionContext(nullptr),
        m_PreProcessStream(nullptr),
        m_InferStream(nullptr),
        m_BufferCopyStream(nullptr),
        m_MeanDataBuffer(nullptr),
        m_Batches(NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE),
        m_InputConsumedEvent(nullptr),
        m_PreProcessCompleteEvent(nullptr),
        m_InferCompleteEvent(nullptr),
        m_LoggingFunc(nullptr),
        m_UserCtx(nullptr),
        m_Initialized(false)
{
    m_Logger.handle = this;
}

/* The function performs all the initialization steps required by the inference
 * engine. */
NvDsInferStatus
NvDsInferContextImpl::initialize(NvDsInferContextInitParams &initParams,
        void *userCtx, NvDsInferContextLoggingFunc logFunc)
{
    cudaError_t cudaReturn;
    bool generateModel = true;
    std::string nvtx_name;

    m_LoggingFunc = logFunc;
    m_UserCtx = userCtx;

    /* Synchronization using once_flag and call_once to ensure TensorRT plugin
     * initialization function is called only once in case of multiple instances
     * of this constructor being called from different threads. */
    {
        static once_flag pluginInitFlag;
        call_once(pluginInitFlag,
                [this]() { initLibNvInferPlugins(&this->m_Logger, ""); } );
    }

    m_UniqueID = initParams.uniqueID;
    m_MaxBatchSize = initParams.maxBatchSize;
    m_NetworkScaleFactor = initParams.networkScaleFactor;
    m_NetworkInputFormat = initParams.networkInputFormat;
    m_NetworkType = initParams.networkType;
    m_UseDBScan = initParams.useDBScan;

    m_ClassifierThreshold = initParams.classifierThreshold;
    m_SegmentationThreshold = initParams.segmentationThreshold;
    m_GpuID = initParams.gpuID;
    m_CopyInputToHostBuffers = initParams.copyInputToHostBuffers;
    m_OutputBufferPoolSize = initParams.outputBufferPoolSize;
    m_Batches.resize(m_OutputBufferPoolSize);

    if (m_UniqueID == 0)
    {
        printError("Unique ID not set");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (m_MaxBatchSize > NVDSINFER_MAX_BATCH_SIZE)
    {
        printError ("Batch-size (%d) more than maximum allowed batch-size (%d)",
                initParams.maxBatchSize, NVDSINFER_MAX_BATCH_SIZE);
        return NVDSINFER_CONFIG_FAILED;
    }

    if (initParams.numOutputLayers > 0 && initParams.outputLayerNames == nullptr)
    {
        printError("NumOutputLayers > 0 but outputLayerNames array not specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    switch (m_NetworkType)
    {
        case NvDsInferNetworkType_Detector:
            m_NumDetectedClasses = initParams.numDetectedClasses;
            if (initParams.numDetectedClasses > 0 &&  initParams.perClassDetectionParams == nullptr)
            {
                printError("NumDetectedClasses > 0 but PerClassDetectionParams array not specified");
                return NVDSINFER_CONFIG_FAILED;
            }

            m_PerClassDetectionParams.assign(initParams.perClassDetectionParams,
                    initParams.perClassDetectionParams + m_NumDetectedClasses);
            m_DetectionParams.numClassesConfigured = initParams.numDetectedClasses;
            m_DetectionParams.perClassThreshold.resize(initParams.numDetectedClasses);

            /* Resize the per class vector to the number of detected classes. */
            m_PerClassObjectList.resize(initParams.numDetectedClasses);
            if (!m_UseDBScan)
            {
                m_PerClassCvRectList.resize(initParams.numDetectedClasses);
            }

            /* Fill the class thresholds in the m_DetectionParams structure. This
             * will be required during parsing. */
            for (unsigned int i = 0; i < initParams.numDetectedClasses; i++)
            {
                m_DetectionParams.perClassThreshold[i] =
                    m_PerClassDetectionParams[i].threshold;
            }
            break;
        case NvDsInferNetworkType_Classifier:
            break;
        case NvDsInferNetworkType_Segmentation:
            break;
        case NvDsInferNetworkType_Other:
            break;
        default:
            printError("Unsupported network type");
            return NVDSINFER_CONFIG_FAILED;
    }

    switch (initParams.networkMode)
    {
        case NvDsInferNetworkMode_FP32:
        case NvDsInferNetworkMode_FP16:
        case NvDsInferNetworkMode_INT8:
            break;
        default:
            printError("Unsupported network dataType");
            return NVDSINFER_CONFIG_FAILED;
    }

    if (m_OutputBufferPoolSize < NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE)
    {
        printError("Output buffer pool size (%d) less than minimum required(%d)",
                m_OutputBufferPoolSize, NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE);
        return NVDSINFER_CONFIG_FAILED;
    }

    /* Set the cuda device to be used. */
    cudaReturn = cudaSetDevice(m_GpuID);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to set cuda device (%s).", cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    /* Create the API root class. */
    m_InferRuntime = createInferRuntime(m_Logger);
    if (!m_InferRuntime)
    {
        printError("Failed to create Infer runtime engine.");
        return NVDSINFER_TENSORRT_ERROR;
    }

    /* Load the custom library if specified. */
    if (!string_empty(initParams.customLibPath))
    {
        m_CustomLibHandle = dlopen (initParams.customLibPath, RTLD_LAZY);
        if (!m_CustomLibHandle)
        {
            printError("Could not open custom lib: %s", dlerror());
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    }

    /* If the custom library is specified, check if PluginFactory instance is
     * required during deserialization of cuda engine. */
    NvDsInferPluginFactoryRuntimeGetFcn fcn = nullptr;
    if (m_CustomLibHandle)
    {
        fcn = (NvDsInferPluginFactoryRuntimeGetFcn)
            dlsym(m_CustomLibHandle, "NvDsInferPluginFactoryRuntimeGet");
        if (fcn)
        {
            if (!fcn(m_RuntimePluginFactory))
            {
                printError("Failed to get runtime plugin factory instance"
                    " from custom library.");
                return NVDSINFER_CUSTOM_LIB_FAILED;
            }
        }
    }

    if (!string_empty(initParams.modelEngineFilePath))
    {
        if (useEngineFile(initParams) == NVDSINFER_SUCCESS)
        {
            generateModel = false;
        }
    }

    if (generateModel)
    {
        NvDsInferStatus status;
        IHostMemory *gieModelStream;
        printInfo("Trying to create engine from model files");

        /* Create the gie Model stream from the model files and other parameters. */
        status = generateTRTModel(initParams, gieModelStream);
        if (status != NVDSINFER_SUCCESS)
        {
            printError("Failed to create engine from model files");
            return status;
        }

        /* Use DLA if specified. */
        if (initParams.useDLA)
        {
            m_InferRuntime->setDLACore(initParams.dlaCore);
        }

        /* Create the cuda engine from the serialized stream. */
        m_CudaEngine =
            m_InferRuntime->deserializeCudaEngine(gieModelStream->data(),
                                                   gieModelStream->size(),
                                                   m_RuntimePluginFactory);
        /* Destroy the model stream, since cuda engine has been serialized. */
        gieModelStream->destroy();

        if (!m_CudaEngine)
        {
            printError("Failed to create engine from serialized stream");
            return NVDSINFER_TENSORRT_ERROR;
        }
        if (checkEngineParams(initParams) !=  NVDSINFER_SUCCESS)
        {
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    m_DlaEnabled = initParams.useDLA;

    /* Get the network input dimensions. */
    DimsCHW inputDims =
        static_cast<DimsCHW&&>(m_CudaEngine->getBindingDimensions(INPUT_LAYER_INDEX));
    m_NetworkInfo.width = inputDims.w();
    m_NetworkInfo.height = inputDims.h();
    m_NetworkInfo.channels = inputDims.c();

    switch (m_NetworkInputFormat)
    {
        case NvDsInferFormat_RGB:
        case NvDsInferFormat_BGR:
            if (m_NetworkInfo.channels != 3)
            {
                printError("RGB/BGR input format specified but network input"
                    " channels is not 3");
                return NVDSINFER_CONFIG_FAILED;
            }
            break;
        case NvDsInferFormat_GRAY:
            if (m_NetworkInfo.channels != 1)
            {
                printError("GRAY input format specified but network input "
                    "channels is not 1.");
                return NVDSINFER_CONFIG_FAILED;
            }
            break;
        default:
            printError("Unknown input format");
            return NVDSINFER_CONFIG_FAILED;
    }

    /* Create the mean data buffer from mean image file or per color component
     * offsets if either are specified. */
    if (!string_empty(initParams.meanImageFilePath) || initParams.numOffsets > 0)
    {
        /* Mean Image File specified. Allocate the mean image buffer on device
         * memory. */
        cudaReturn = cudaMalloc((void **)&m_MeanDataBuffer,
                                 m_NetworkInfo.width * m_NetworkInfo.height *
                                 m_NetworkInfo.channels * sizeof (float));
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to allocate cuda buffer for mean image(%s)",
                    cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }
        /* Read the mean image file (PPM format) if specified and copy the
         * contents into the buffer. */
        if (!string_empty(initParams.meanImageFilePath))
        {
            if (!file_accessible(initParams.meanImageFilePath))
            {
                printError("Cannot access mean image file '%s'",
                        initParams.meanImageFilePath);
                return NVDSINFER_CONFIG_FAILED;
            }
            NvDsInferStatus status = readMeanImageFile(initParams.meanImageFilePath);
            if (status != NVDSINFER_SUCCESS)
            {
                printError("Failed to read mean image file");
                return status;
            }
        }
        /* Create the mean data buffer from per-channel offsets. */
        else
        {
            /* Make sure the number of offsets are equal to the number of input
             * channels. */
            if (initParams.numOffsets != m_NetworkInfo.channels)
            {
                printError("Number of offsets(%d) not equal to number of input "
                        "channels(%d)", initParams.numOffsets,
                        m_NetworkInfo.channels);
                return NVDSINFER_CONFIG_FAILED;
            }

            vector<float> meanData(
                    m_NetworkInfo.channels * m_NetworkInfo.width *
                    m_NetworkInfo.height);
            for (size_t j = 0; j < m_NetworkInfo.width * m_NetworkInfo.height; j++)
            {
                for (size_t i = 0; i < m_NetworkInfo.channels; i++)
                {
                    meanData[j * m_NetworkInfo.channels + i] = initParams.offsets[i];
                }
            }
            cudaReturn = cudaMemcpy(m_MeanDataBuffer, meanData.data(),
                    meanData.size() * sizeof(float), cudaMemcpyHostToDevice);
            if (cudaReturn != cudaSuccess)
            {
                printError("Failed to copy mean data to mean data cuda buffer(%s)",
                        cudaGetErrorName(cudaReturn));
                return NVDSINFER_CUDA_ERROR;
            }
        }
    }

    /* Get information on all bound layers. */
    getBoundLayersInfo();

    /* Create the Infer Execution Context. */
    m_InferExecutionContext = m_CudaEngine->createExecutionContext();
    if (!m_InferExecutionContext)
    {
        printError("Failed to create Infer Execution Context");
        return NVDSINFER_TENSORRT_ERROR;
    }

    /* Create the cuda stream on which pre-processing jobs will be executed. */
    cudaReturn = cudaStreamCreateWithFlags(&m_PreProcessStream,
            cudaStreamNonBlocking);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to create cudaStream(%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_TENSORRT_ERROR;
    }
    nvtx_name = "nvdsinfer_preprocess_uid=" + to_string(m_UniqueID);
    nvtxNameCudaStreamA (m_PreProcessStream, nvtx_name.c_str());

    /* Create the cuda stream on which inference jobs will be executed. */
    cudaReturn = cudaStreamCreateWithFlags(&m_InferStream, cudaStreamNonBlocking);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to create cudaStream(%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_infer_uid=" + to_string(m_UniqueID);
    nvtxNameCudaStreamA (m_InferStream, nvtx_name.c_str());

    /* Create the cuda stream on which device to host memcpy jobs will be
     * executed. */
    cudaReturn = cudaStreamCreateWithFlags (&m_BufferCopyStream,
            cudaStreamNonBlocking);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to create cudaStream(%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_DtoHcopy_uid=" + to_string(m_UniqueID);
    nvtxNameCudaStreamA (m_BufferCopyStream, nvtx_name.c_str());

    /* Allocate binding buffers on the device and the corresponding host
     * buffers. */
    NvDsInferStatus status = allocateBuffers();
    if (status != NVDSINFER_SUCCESS)
    {
        printError("Failed to allocate buffers");
        return status;
    }

    /* Parse the labels file if specified. */
    if (!string_empty(initParams.labelsFilePath))
    {
        if (!file_accessible(initParams.labelsFilePath))
        {
            printError("Could not access labels file '%s'", initParams.labelsFilePath);
            return NVDSINFER_CONFIG_FAILED;
        }
        NvDsInferStatus status = parseLabelsFile(initParams.labelsFilePath);
        if (status != NVDSINFER_SUCCESS)
        {
            printError("Failed to read labels file");
            return status;
        }
    }

    /* Cuda event to synchronize between consumption of input binding buffer by
     * the cuda engine and the pre-processing kernel which writes to the input
     * binding buffer. */
    cudaReturn = cudaEventCreateWithFlags(&m_InputConsumedEvent,
            cudaEventDisableTiming);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to create cuda event(%s)", cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_TRT_input_consumed_uid=" + to_string(m_UniqueID);
    nvtxNameCudaEventA (m_InputConsumedEvent, nvtx_name.c_str());

    /* Cuda event to synchronize between completion of the pre-processing kernels
     * and enqueuing the next set of binding buffers for inference. */
    cudaReturn = cudaEventCreateWithFlags(&m_PreProcessCompleteEvent,
            cudaEventDisableTiming);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to create cuda event(%s)", cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_preprocess_complete_uid=" + to_string(m_UniqueID);
    nvtxNameCudaEventA (m_PreProcessCompleteEvent, nvtx_name.c_str());

    /* Cuda event to synchronize between completion of inference on a batch
     * and copying the output contents from device to host memory. */
    cudaReturn = cudaEventCreateWithFlags(&m_InferCompleteEvent,
            cudaEventDisableTiming);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to create cuda event(%s)", cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }
    nvtx_name = "nvdsinfer_infer_complete_uid=" + to_string(m_UniqueID);
    nvtxNameCudaEventA (m_InferCompleteEvent, nvtx_name.c_str());

    /* If custom parse function is specified get the function address from the
     * custom library. */
    if (m_CustomLibHandle && m_NetworkType == NvDsInferNetworkType_Detector &&
            !string_empty(initParams.customBBoxParseFuncName))
    {
        m_CustomBBoxParseFunc =
            (NvDsInferParseCustomFunc) dlsym(m_CustomLibHandle,
                    initParams.customBBoxParseFuncName);
        if (!m_CustomBBoxParseFunc)
        {
            printError("Could not find parse func '%s' in custom library",
                initParams.customBBoxParseFuncName);
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    if (m_CustomLibHandle && m_NetworkType == NvDsInferNetworkType_Classifier &&
            !string_empty(initParams.customClassifierParseFuncName))
    {
        m_CustomClassifierParseFunc =
            (NvDsInferClassiferParseCustomFunc) dlsym(m_CustomLibHandle,
                    initParams.customClassifierParseFuncName);
        if (!m_CustomClassifierParseFunc)
        {
            printError("Could not find parse func '%s' in custom library",
                initParams.customClassifierParseFuncName);
            return NVDSINFER_CONFIG_FAILED;
        }
    }

    /* If there are more than one input layers (non-image input) and custom
     * library is specified, try to initialize these layers. */
    if (m_AllLayerInfo.size() > 1 + m_OutputLayerInfo.size())
    {
        NvDsInferStatus status = initNonImageInputLayers();
        if (status != NVDSINFER_SUCCESS)
        {
            printError("Failed to initialize non-image input layers");
            return status;
        }
    }

    if (m_UseDBScan)
    {
        m_DBScanHandle = NvDsInferDBScanCreate();
    }

    m_Initialized = true;

    return NVDSINFER_SUCCESS;
}

/* Get the network input resolution. This is required since this implementation
 * requires that the caller supplies an input buffer having the network
 * resolution.
 */
void
NvDsInferContextImpl::getNetworkInfo(NvDsInferNetworkInfo &networkInfo)
{
    networkInfo = m_NetworkInfo;
}

/* Allocate binding buffers for all bound layers on the device memory. The size
 * of the buffers allocated is calculated from the dimensions of the layers, the
 * data type of the layer and the max batch size of the infer cuda engine.
 *
 * NvDsInfer enqueue API requires an array of (void *) buffer pointers. The length
 * of the array is equal to the number of bound layers. The buffer corresponding
 * to a layer is placed at an index equal to the layer's binding index.
 *
 * Also allocate corresponding host buffers for output layers in system memory.
 *
 * Multiple sets of the device and host buffers are allocated so that (inference +
 * device to host copy) and output layers parsing can be parallelized.
 */
NvDsInferStatus
NvDsInferContextImpl::allocateBuffers()
{
    cudaError_t cudaReturn;

//    m_CudaEngine->createExecutionContext();
    /* Resize the binding buffers vector to the number of bound layers. */
    m_BindingBuffers.assign(m_AllLayerInfo.size(), nullptr);

    for (unsigned int i = 0; i < m_AllLayerInfo.size(); i++)
    {
        size_t size = m_MaxBatchSize * m_AllLayerInfo[i].dims.numElements *
            getElementSize(m_AllLayerInfo[i].dataType);

        /* Do not allocate device memory for output layers here. */
        if (!m_CudaEngine->bindingIsInput(i))
            continue;

        /* Allocate device memory for the binding buffer. */
        cudaReturn = cudaMalloc(&m_BindingBuffers[i], size);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to allocate cuda buffer(%s)",
                    cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }
    }

    /* Initialize the batch vector, allocate host memory for the layers,
     * add all the free indexes to the free queue. */
    for (unsigned int i = 0; i < m_Batches.size(); i++)
    {
        NvDsInferBatch & batch = m_Batches[i];
        /* Resize the host buffers vector to the number of bound layers. */
        batch.m_HostBuffers.resize(m_AllLayerInfo.size());
        batch.m_DeviceBuffers.assign(m_AllLayerInfo.size(), nullptr);


        for (unsigned int j = 0; j < m_AllLayerInfo.size(); j++)
        {
            size_t size = m_MaxBatchSize * m_AllLayerInfo[j].dims.numElements *
                getElementSize(m_AllLayerInfo[j].dataType);

            if (m_CudaEngine->bindingIsInput(j))
            {
                /* Reuse input binding buffer pointers. */
                batch.m_DeviceBuffers[j] = m_BindingBuffers[j];
            }
            else
            {
                /* Allocate device memory for output layers here. */
                cudaReturn = cudaMalloc(&batch.m_DeviceBuffers[j], size);
                if (cudaReturn != cudaSuccess)
                {
                    printError("Failed to allocate cuda buffer(%s)",
                            cudaGetErrorName(cudaReturn));
                    return NVDSINFER_CUDA_ERROR;
                }
            }

            /* Allocate host memory for input layers only if application
             * needs access to the input layer contents. */
            if (m_CudaEngine->bindingIsInput(j) && !m_CopyInputToHostBuffers)
                continue;

            /* Resize the uint8_t vector to the size (in bytes) of the buffer.
             * The underlying heap memory can be used as host buffer. */
            batch.m_HostBuffers[j].resize(size);
        }
        cudaReturn = cudaEventCreateWithFlags (&batch.m_CopyCompleteEvent,
                cudaEventDisableTiming | cudaEventBlockingSync);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to create cuda event(%s)",
                    cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }

        /* Add all the indexes to the free queue initially. */
        m_FreeIndexQueue.push(i);
    }

    return NVDSINFER_SUCCESS;
}

/* Get properties of bound layers like the name, dimension, datatype and
 * fill the m_AllLayerInfo and m_OutputLayerInfo vectors.
 */
NvDsInferStatus
NvDsInferContextImpl::getBoundLayersInfo()
{
    for (int i = 0; i < m_CudaEngine->getNbBindings(); i++)
    {
        NvDsInferLayerInfo info;
        Dims d = m_CudaEngine->getBindingDimensions(i);

        info.isInput = m_CudaEngine->bindingIsInput(i);
        info.bindingIndex = i;
        info.layerName = m_CudaEngine->getBindingName(i);
        info.dims.numDims = d.nbDims;
        info.dims.numElements = 1;
        for (int j = 0; j < d.nbDims; j++)
        {
            info.dims.d[j] = d.d[j];
            info.dims.numElements *= d.d[j];
        }

        switch (m_CudaEngine->getBindingDataType(i))
        {
            case DataType::kFLOAT:
                info.dataType = FLOAT;
                break;
            case DataType::kHALF:
                info.dataType = HALF;
                break;
            case DataType::kINT32:
                info.dataType = INT32;
                break;
            case DataType::kINT8:
                info.dataType = INT8;
                break;
            default:
                printError("Unknown data type for bound layer i(%s)",
                        info.layerName);
                return NVDSINFER_TENSORRT_ERROR;
        }

        m_AllLayerInfo.push_back(info);
        if (!m_CudaEngine->bindingIsInput(i))
            m_OutputLayerInfo.push_back(info);
    }
    return NVDSINFER_SUCCESS;
}

/* Initialize non-image input layers if the custom library has implemented
 * the interface. */
NvDsInferStatus
NvDsInferContextImpl::initNonImageInputLayers()
{
    cudaError_t cudaReturn;

    /* Needs the custom library to be specified. */
    if (m_CustomLibHandle == nullptr)
    {
        printWarning("More than one input layers but custom initialization "
            "function not implemented");
        return NVDSINFER_SUCCESS;
    }

    /* Check if the interface to initialize the layers has been implemented. */
    NvDsInferInitializeInputLayersFcn fcn = (NvDsInferInitializeInputLayersFcn)
        dlsym(m_CustomLibHandle, "NvDsInferInitializeInputLayers");
    if (fcn == nullptr)
    {
        printWarning("More than one input layers but custom initialization "
            "function not implemented");
        return NVDSINFER_SUCCESS;
    }

    /* Interface implemented.  */
    /* Vector of NvDsInferLayerInfo for non-image input layers. */
    vector<NvDsInferLayerInfo> inputLayers;
    for (auto &layer : m_AllLayerInfo)
    {
        if (m_CudaEngine->bindingIsInput(layer.bindingIndex) &&
                layer.bindingIndex != INPUT_LAYER_INDEX)
        {
            inputLayers.push_back(layer);
        }
    }

    /* Vector of host memories that can be initialized using CPUs. */
    vector<std::vector<uint8_t>> initBuffers(inputLayers.size());

    for (size_t i = 0; i < inputLayers.size(); i++)
    {
        /* For each layer calculate the size required for the layer, allocate
         * the host memory and assign the pointer to layer info structure. */
        size_t size = inputLayers[i].dims.numElements *
            getElementSize(inputLayers[i].dataType) * m_MaxBatchSize;
        initBuffers[i].resize(size);
        inputLayers[i].buffer = (void *) initBuffers[i].data();
    }

    /* Call the input layer initialization function. */
    if (!fcn(inputLayers, m_NetworkInfo, m_MaxBatchSize))
    {
        printError("Failed to initialize input layers using "
                "NvDsInferInitializeInputLayers() in custom lib");
        return NVDSINFER_CUSTOM_LIB_FAILED;
    }

    /* Memcpy the initialized contents from the host memory to device memory for
     * layer binding buffers. */
    for (size_t i = 0; i < inputLayers.size(); i++)
    {
        cudaReturn = cudaMemcpyAsync(m_BindingBuffers[inputLayers[i].bindingIndex],
                initBuffers[i].data(), initBuffers[i].size(),
                cudaMemcpyHostToDevice, m_InferStream);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to copy from host to device memory (%s)",
                    cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }
        /* Application has requested access to the bound buffer contents. Copy
         * the contents to all sets of host buffers. */
        if (m_CopyInputToHostBuffers)
        {
            for (size_t j = 0; j < m_Batches.size(); j++)
            {
                for (size_t i = 0; i < inputLayers.size(); i++)
                {
                    m_Batches[j].m_HostBuffers[inputLayers[i].bindingIndex].
                        assign(initBuffers[i].begin(), initBuffers[i].end());
                }
            }
        }
    }
    cudaReturn = cudaStreamSynchronize(m_InferStream);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to synchronize cuda stream(%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

/* Parse the labels file and extract the class label strings. For format of
 * the labels file, please refer to the custom models section in the DeepStreamSDK
 * documentation.
 */
NvDsInferStatus
NvDsInferContextImpl::parseLabelsFile(char *labelsFilePath)
{
    ifstream labels_file(labelsFilePath);
    string delim { ';' };
    while (!labels_file.eof())
    {
        string line, word;
        vector<string> l;
        size_t pos = 0, oldpos = 0;

        getline(labels_file, line, '\n');
        if (line.empty())
            continue;

        while ((pos = line.find(delim, oldpos)) != string::npos)
        {
            word = line.substr(oldpos, pos - oldpos);
            l.push_back(word);
            oldpos = pos + delim.length();
        }
        l.push_back(line.substr(oldpos));
        m_Labels.push_back(l);
    }
    return NVDSINFER_SUCCESS;
}

/* Read the mean image ppm file and copy the mean image data to the mean
 * data buffer allocated on the device memory.
 */
NvDsInferStatus
NvDsInferContextImpl::readMeanImageFile(char *meanImageFilePath)
{
    ifstream infile(meanImageFilePath, std::ifstream::binary);
    size_t size = m_NetworkInfo.width * m_NetworkInfo.height *
        m_NetworkInfo.channels;
    char tempMeanDataChar[size];
    float tempMeanDataFloat[size];
    cudaError_t cudaReturn;

    if (!infile.good())
    {
        printError("Could not open mean image file '%s'", meanImageFilePath);
        return NVDSINFER_CONFIG_FAILED;
    }

    string magic, max;
    unsigned int h, w;
    infile >> magic >> h >> w >> max;

    if (magic != "P3" && magic != "P6")
    {
        printError("Magic PPM identifier check failed");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (w != m_NetworkInfo.width || h != m_NetworkInfo.height)
    {
        printError("Mismatch between ppm mean image resolution(%d x %d) and "
                "network resolution(%d x %d)", w, h, m_NetworkInfo.width,
                m_NetworkInfo.height);
        return NVDSINFER_CONFIG_FAILED;
    }

    infile.get();
    infile.read(tempMeanDataChar, size);
    if (infile.gcount() != (int) size)
    {
        printError("Failed to read sufficient bytes from mean file");
        return NVDSINFER_CONFIG_FAILED;
    }

    for (size_t i = 0; i < size; i++)
    {
        tempMeanDataFloat[i] = (float) tempMeanDataChar[i];
    }

    cudaReturn = cudaMemcpy(m_MeanDataBuffer, tempMeanDataFloat,
                            size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to copy mean data to mean data buffer (%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::queueInputBatch(NvDsInferContextBatchInput &batchInput)

{
    unsigned int batchSize = batchInput.numInputFrames;
    unsigned int batchIndex;
    void *bindingBuffers[m_AllLayerInfo.size()];
    NvDsInferStatus status;
    NvDsInferConvertFcn convertFcn = nullptr;

    /* Check that current batch size does not exceed max batch size. */
    if (batchSize > m_MaxBatchSize)
    {
        printError("Not inferring on batch since it's size(%d) exceeds max batch"
                " size(%d)", batchSize, m_MaxBatchSize);
        return NVDSINFER_INVALID_PARAMS;
    }

    /* DLA does not allow enqueuing batches smaller than the engine's maxBatchSize. */
    int enqueueBatchSize = m_DlaEnabled ? m_MaxBatchSize : batchSize;

    /* Set the cuda device to be used. */
    cudaError_t cudaReturn = cudaSetDevice(m_GpuID);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to set cuda device(%s)", cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }


    /* Make the future jobs on the stream wait till the infer engine consumes
     * the previous contents of the input binding buffer. */
    cudaReturn = cudaStreamWaitEvent (m_PreProcessStream, m_InputConsumedEvent, 0);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to make stream wait on event(%s)",
                cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    /* Find the required conversion function. */
    switch (m_NetworkInputFormat)
    {
        case NvDsInferFormat_RGB:
            switch (batchInput.inputFormat)
            {
                case NvDsInferFormat_RGB:
                    convertFcn = NvDsInferConvert_C3ToP3Float;
                    break;
                case NvDsInferFormat_BGR:
                    convertFcn = NvDsInferConvert_C3ToP3RFloat;
                    break;
                case NvDsInferFormat_RGBA:
                    convertFcn = NvDsInferConvert_C4ToP3Float;
                    break;
                case NvDsInferFormat_BGRx:
                    convertFcn = NvDsInferConvert_C4ToP3RFloat;
                    break;
                default:
                    printError("Input format conversion is not supported");
                    return NVDSINFER_INVALID_PARAMS;
            }
            break;
        case NvDsInferFormat_BGR:
            switch (batchInput.inputFormat)
            {
                case NvDsInferFormat_RGB:
                    convertFcn = NvDsInferConvert_C3ToP3RFloat;
                    break;
                case NvDsInferFormat_BGR:
                    convertFcn = NvDsInferConvert_C3ToP3Float;
                    break;
                case NvDsInferFormat_RGBA:
                    convertFcn = NvDsInferConvert_C4ToP3RFloat;
                    break;
                case NvDsInferFormat_BGRx:
                    convertFcn = NvDsInferConvert_C4ToP3Float;
                    break;
                default:
                    printError("Input format conversion is not supported");
                    return NVDSINFER_INVALID_PARAMS;
            }
            break;
        case NvDsInferFormat_GRAY:
            if (batchInput.inputFormat != NvDsInferFormat_GRAY)
            {
                printError("Input frame format is not GRAY.");
                return NVDSINFER_INVALID_PARAMS;
            }
            convertFcn = NvDsInferConvert_C1ToP1Float;
            break;
        default:
            printError("Unsupported network input format");
            return NVDSINFER_INVALID_PARAMS;
    }

    /* For each frame in the input batch convert/copy to the input binding buffer. */
    for (unsigned int i = 0; i < batchSize; i++)
    {
        float *outPtr = (float *) m_BindingBuffers[INPUT_LAYER_INDEX] +
            i * m_AllLayerInfo[INPUT_LAYER_INDEX].dims.numElements;

        /* Input needs to be pre-processed. */
        convertFcn(outPtr, (unsigned char*) batchInput.inputFrames[i],
                m_NetworkInfo.width, m_NetworkInfo.height,
                batchInput.inputPitch, m_NetworkScaleFactor,
                m_MeanDataBuffer, m_PreProcessStream);
    }

    /* We may use multiple sets of the output device and host buffers since while the
     * output of one batch is being parsed on the CPU, we can queue
     * pre-processing and inference of another on the GPU. Pop an index from the
     * free queue. Wait if queue is empty. */
    {
        unique_lock<mutex> lock(m_QueueMutex);
        while (m_FreeIndexQueue.empty())
        {
            m_QueueCondition.wait(lock);
        }
        batchIndex = m_FreeIndexQueue.front();
        m_FreeIndexQueue.pop();
    }

    /* Inputs can be returned back once pre-processing is complete. */
    if (batchInput.returnInputFunc)
    {
        cudaReturn = cudaStreamAddCallback(m_PreProcessStream, returnInputCudaCallback,
                new NvDsInferReturnInputPair(batchInput.returnInputFunc,
                    batchInput.returnFuncData), 0);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to add cudaStream callback for returning input buffers (%s)",
                    cudaGetErrorName(cudaReturn));
            return NVDSINFER_CUDA_ERROR;
        }
    }

    /* Fill the array of binding buffers for the current batch. */
    std::copy(m_Batches[batchIndex].m_DeviceBuffers.begin(),
            m_Batches[batchIndex].m_DeviceBuffers.end(), bindingBuffers);

    /* Record CUDA event to synchronize the completion of pre-processing kernels. */
    cudaReturn = cudaEventRecord(m_PreProcessCompleteEvent, m_PreProcessStream);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to record cuda event (%s)",
                cudaGetErrorName(cudaReturn));
        status = NVDSINFER_CUDA_ERROR;
        goto error;
    }

    /* Make the future jobs on the stream wait till pre-processing kernels finish. */
    cudaReturn = cudaStreamWaitEvent (m_InferStream, m_PreProcessCompleteEvent, 0);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to make stream wait on event(%s)",
                cudaGetErrorName(cudaReturn));
        status = NVDSINFER_CUDA_ERROR;
        goto error;
    }

    {
        std::unique_lock<std::mutex> deferLock(DlaExecutionMutex, std::defer_lock);

        /* IExecutionContext::enqueue is not thread safe in case of DLA */
        if (m_DlaEnabled)
            deferLock.lock();

        /* Queue the bound buffers for inferencing. */
        if (!m_InferExecutionContext->enqueue(enqueueBatchSize, bindingBuffers,
                                              m_InferStream, &m_InputConsumedEvent))
        {
            printError("Failed to enqueue inference batch");
            status = NVDSINFER_TENSORRT_ERROR;
            goto error;
        }
    }

    /* Record event on m_InferStream to indicate completion of inference on the
     * current batch. */
    cudaReturn = cudaEventRecord (m_InferCompleteEvent, m_InferStream);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to record cuda event (%s)", cudaGetErrorName(cudaReturn));
        status = NVDSINFER_CUDA_ERROR;
        goto error;
    }

    /* Make future copy jobs on the buffer copy stream wait on the infer
     * completion event. */
    cudaReturn = cudaStreamWaitEvent (m_BufferCopyStream, m_InferCompleteEvent, 0);
    if (cudaReturn != cudaSuccess)
    {
        printError("CUDA Stream failed to wait on event (%s)",
                cudaGetErrorName(cudaReturn));
        status = NVDSINFER_CUDA_ERROR;
        goto error;
    }

    /* Queue the copy of output contents from device to host memory after the
     * infer completion event. */
    {
        NvDsInferBatch &batch = m_Batches[batchIndex];
        batch.m_BatchSize = batchSize;

        for (unsigned int i = 0; i < m_OutputLayerInfo.size(); i++)
        {
            NvDsInferLayerInfo & info = m_OutputLayerInfo[i];
            cudaReturn =
                cudaMemcpyAsync(batch.m_HostBuffers[info.bindingIndex].data(),
                                 batch.m_DeviceBuffers[info.bindingIndex],
                                 getElementSize(info.dataType) *
                                 info.dims.numElements * batch.m_BatchSize,
                                 cudaMemcpyDeviceToHost, m_BufferCopyStream);
            if (cudaReturn != cudaSuccess)
            {
                printError("cudaMemcpyAsync for output buffers failed (%s)",
                        cudaGetErrorName(cudaReturn));
                status = NVDSINFER_CUDA_ERROR;
                goto error;
            }
        }
        if (m_CopyInputToHostBuffers)
        {
            NvDsInferLayerInfo &info = m_AllLayerInfo[INPUT_LAYER_INDEX];
            cudaReturn =
                cudaMemcpyAsync(batch.m_HostBuffers[info.bindingIndex].data(),
                    m_BindingBuffers[info.bindingIndex],
                    getElementSize(info.dataType) *
                    info.dims.numElements * batch.m_BatchSize,
                    cudaMemcpyDeviceToHost, m_BufferCopyStream);
            if (cudaReturn != cudaSuccess)
            {
                printError("cudaMemcpyAsync for input layer failed (%s)",
                        cudaGetErrorName(cudaReturn));
                status = NVDSINFER_CUDA_ERROR;
                goto error;
            }
        }
        /* Record CUDA event to later synchronize for the copy to actually
         * complete. */
        cudaReturn = cudaEventRecord(batch.m_CopyCompleteEvent,
                m_BufferCopyStream);
        if (cudaReturn != cudaSuccess)
        {
            printError("Failed to record cuda event (%s)",
                    cudaGetErrorName(cudaReturn));
            status = NVDSINFER_CUDA_ERROR;
            goto error;
        }
    }

    /* Push the batch index into the processing queue. */
    {
        unique_lock<mutex> lock(m_QueueMutex);
        m_ProcessIndexQueue.push(batchIndex);
        m_QueueCondition.notify_one();
    }
    return NVDSINFER_SUCCESS;

error:
    {
        unique_lock<mutex> lock(m_QueueMutex);
        m_FreeIndexQueue.push(batchIndex);
    }
    return status;
}

/* Dequeue batch output of the inference engine for each batch input. */
NvDsInferStatus
NvDsInferContextImpl::dequeueOutputBatch(NvDsInferContextBatchOutput &batchOutput)
{
    unsigned int batchIndex;

    /* Set the cuda device */
    cudaError_t cudaReturn = cudaSetDevice(m_GpuID);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to set cuda device (%s)", cudaGetErrorName(cudaReturn));
        return NVDSINFER_CUDA_ERROR;
    }

    /* Pop a batch index from the process queue. Wait if
     * the queue is empty. */
    {
        unique_lock<mutex> lock(m_QueueMutex);
        while (m_ProcessIndexQueue.empty())
        {
            m_QueueCondition.wait(lock);
        }
        batchIndex = m_ProcessIndexQueue.front();
        m_ProcessIndexQueue.pop();
    }
    NvDsInferBatch & batch = m_Batches[batchIndex];

    /* Wait for the copy to the current set of host buffers to complete. */
    cudaReturn = cudaEventSynchronize (batch.m_CopyCompleteEvent);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to synchronize on cuda event (%s)",
                cudaGetErrorName(cudaReturn));
        {
            unique_lock<std::mutex> lock(m_QueueMutex);
            m_FreeIndexQueue.push(batchIndex);
            m_QueueCondition.notify_one();
        }
        return NVDSINFER_CUDA_ERROR;
    }

    batchOutput.frames = new NvDsInferFrameOutput[batch.m_BatchSize];
    batchOutput.numFrames = batch.m_BatchSize;
    /* For each frame in the current batch, parse the output and add the frame
     * output to the batch output. The number of frames output in one batch
     * will be equal to the number of frames present in the batch during queuing
     * at the input.
     */
    for (unsigned int index = 0; index < batch.m_BatchSize; index++)
    {
        NvDsInferFrameOutput &frameOutput = batchOutput.frames[index];
        frameOutput.outputType = NvDsInferNetworkType_Other;

        /* Calculate the pointer to the output for each frame in the batch for
         * each output layer buffer. The NvDsInferLayerInfo vector for output
         * layers is passed to the output parsing function. */
        for (unsigned int i = 0; i < m_OutputLayerInfo.size(); i++)
        {
            NvDsInferLayerInfo & info = m_OutputLayerInfo[i];
            info.buffer =
                (void *)(batch.m_HostBuffers[info.bindingIndex].data() +
                         info.dims.numElements *
                         getElementSize(info.dataType) * index);
        }

        switch (m_NetworkType)
        {
            case NvDsInferNetworkType_Detector:
                fillDetectionOutput(frameOutput.detectionOutput);
                frameOutput.outputType = NvDsInferNetworkType_Detector;
                break;
            case NvDsInferNetworkType_Classifier:
                fillClassificationOutput(frameOutput.classificationOutput);
                frameOutput.outputType = NvDsInferNetworkType_Classifier;
                break;
            case NvDsInferNetworkType_Segmentation:
                fillSegmentationOutput(frameOutput.segmentationOutput);
                frameOutput.outputType = NvDsInferNetworkType_Segmentation;
                break;
            default:
                break;
        }
    }

    /* Fill the host buffers information in the output. */
    batchOutput.outputBatchID = batchIndex;
    batchOutput.numHostBuffers = m_AllLayerInfo.size();
    batchOutput.hostBuffers = new void*[m_AllLayerInfo.size()];
    for (size_t i = 0; i < batchOutput.numHostBuffers; i++)
    {
        batchOutput.hostBuffers[i] = m_Batches[batchIndex].m_HostBuffers[i].data();
    }

    batchOutput.numOutputDeviceBuffers = m_OutputLayerInfo.size();
    batchOutput.outputDeviceBuffers = new void*[m_OutputLayerInfo.size()];
    for (size_t i = 0; i < batchOutput.numOutputDeviceBuffers; i++)
    {
        batchOutput.outputDeviceBuffers[i] =
            m_Batches[batchIndex].m_DeviceBuffers[m_OutputLayerInfo[i].bindingIndex];
    }

    /* Mark the set of host buffers as not with the context. */
    m_Batches[batchIndex].m_BuffersWithContext = false;
    return NVDSINFER_SUCCESS;
}

/**
 * Release a set of host buffers back to the context.
 */
void
NvDsInferContextImpl::releaseBatchOutput(NvDsInferContextBatchOutput &batchOutput)
{
    unique_lock < std::mutex > lock (m_QueueMutex);
    unsigned int outputBatchID = batchOutput.outputBatchID;

    /* Check for a valid id */
    if (outputBatchID >= m_Batches.size())
    {
        printWarning("Tried to release an unknown outputBatchID");
        return;
    }
    /* And if the batch is not already with the context. */
    if (m_Batches[outputBatchID].m_BuffersWithContext)
    {
        printWarning("Tried to release an outputBatchID which is"
            " already with the context");
        return;
    }
    m_Batches[outputBatchID].m_BuffersWithContext = true;
    m_FreeIndexQueue.push (outputBatchID);
    m_QueueCondition.notify_one ();

    /* Free memory allocated in dequeueOutputBatch */
    for (unsigned int i = 0; i < batchOutput.numFrames; i++)
    {
        releaseFrameOutput(batchOutput.frames[i]);
    }

    delete[] batchOutput.frames;
    delete[] batchOutput.hostBuffers;
    delete[] batchOutput.outputDeviceBuffers;
}

/**
 * Fill all the bound layers information in the vector.
 */
void
NvDsInferContextImpl::fillLayersInfo(vector<NvDsInferLayerInfo> &layersInfo)
{
    layersInfo.assign (m_AllLayerInfo.begin(), m_AllLayerInfo.end());
}

const vector<std::vector<std::string>> &
NvDsInferContextImpl::getLabels()
{
    return m_Labels;
}

/* Check if the runtime cuda engine is compatible with requested configuration. */
NvDsInferStatus
NvDsInferContextImpl::checkEngineParams(NvDsInferContextInitParams &initParams)
{
    /* Check if the cuda engine can support requested max batch size. */
    if ((int) m_MaxBatchSize > m_CudaEngine->getMaxBatchSize())
    {
        printWarning("Requested Max Batch Size is less than engine batch size");
        return NVDSINFER_CONFIG_FAILED;
    }

    for (unsigned int i = 0; i < initParams.numOutputLayers; i++)
    {
        int bindingIndex = m_CudaEngine->getBindingIndex(initParams.outputLayerNames[i]);
        if (bindingIndex == -1 || m_CudaEngine->bindingIsInput(bindingIndex))
        {
            printWarning("Could not find output layer '%s' in engine",
                    initParams.outputLayerNames[i]);
        }
    }

    return NVDSINFER_SUCCESS;
}

/* Try to create the Cuda Engine from a serialized file. */
NvDsInferStatus
NvDsInferContextImpl::useEngineFile(NvDsInferContextInitParams &initParams)
{
    NvDsInferStatus status;
    size_t size = 0;
    size_t i = 0;
    ifstream gieModelFile(initParams.modelEngineFilePath);
    if (!gieModelFile.good())
    {
        printWarning("Failed to read from model engine file");
        return NVDSINFER_CONFIG_FAILED;
    }

    /* Get the engine file size and read contents into a char buffer. */
    gieModelFile.seekg(0, ios::end);
    size = gieModelFile.tellg();
    gieModelFile.seekg(0, ios::beg);

    std::vector<char> buff(size);
    while (gieModelFile.get(buff[i]))
        i++;
    gieModelFile.close();

    /* Use DLA if specified. */
    if (initParams.useDLA)
    {
        m_InferRuntime->setDLACore(initParams.dlaCore);
    }

    /* Create the cuda engine from the serialized engine file contents. */
    m_CudaEngine = m_InferRuntime->deserializeCudaEngine((void *) buff.data(),
            size, m_RuntimePluginFactory);
    if (!m_CudaEngine)
    {
        printWarning("Failed to create engine from file");
        return NVDSINFER_TENSORRT_ERROR;
    }

    /* Check if the deserialized cuda engine is compatible with requested
     * configuration. */
    status = checkEngineParams(initParams);
    if (status != NVDSINFER_SUCCESS)
    {
        /* Cannot use deserialized cuda engine. Destroy the engine. */
        m_CudaEngine->destroy();
        m_CudaEngine = nullptr;
    }
    return status;
}

/* Custom unique_ptr subclass with deleter functions for TensorRT objects. */
template <class T>
class NvDsInferUniquePtr : public std::unique_ptr<T, void (*)(T *)>
{
    public:
        NvDsInferUniquePtr(T * t = nullptr) :
            std::unique_ptr<T, void (*)(T *)>(t, [](T *t){if (t) t->destroy();})
            {}
};

/* Create cudaengine for the model from the init params
 * (caffemodel & prototxt/uff/onnx, int8 calibration tables, etc) and return the
 * serialized cuda engine stream. */
NvDsInferStatus
NvDsInferContextImpl::generateTRTModel(
        NvDsInferContextInitParams &initParams,
        IHostMemory *&gieModelStream)
{
    /* Custom implementation of unique_ptr ensures that corresponding destroy
     * methods of TensorRT objects get called when the pointer variables go out
     * of scope. */
    NvDsInferUniquePtr<IBuilder> builder = nvinfer1::createInferBuilder(m_Logger);
    NvDsInferUniquePtr<INetworkDefinition> network = builder->createNetwork ();
    NvDsInferUniquePtr<ICudaEngine> cudaEngine;

    NvDsInferUniquePtr<nvcaffeparser1::ICaffeParser> caffeParser;
    NvDsInferUniquePtr<nvuffparser::IUffParser> uffParser;
    NvDsInferUniquePtr<nvonnxparser::IParser> onnxParser;

    NvDsInferInt8Calibrator pCalibrator(initParams.int8CalibrationFilePath);
    NvDsInferNetworkMode networkMode = initParams.networkMode;
    DataType modelDataType;

    stringstream engineFileName;

    NvDsInferPluginFactoryCaffe caffePluginFactory{nullptr};
    NvDsInferPluginFactoryUff uffPluginFactory{nullptr};

    NvDsInferCudaEngineGetFcn cudaEngineGetFcn = nullptr;

    switch (networkMode)
    {
        case NvDsInferNetworkMode_FP32:
        case NvDsInferNetworkMode_FP16:
        case NvDsInferNetworkMode_INT8:
            break;
        default:
            printError("Unknown network mode %d", networkMode);
            return NVDSINFER_CONFIG_FAILED;
    }

    if (!string_empty(initParams.tltEncodedModelFilePath))
    {
        /* Use the CUDA engine creation function for TLT encoded models provided
         * by NvDsInferUtils. */
        cudaEngineGetFcn = NvDsInferCudaEngineGetFromTltModel;
    }
    else if (m_CustomLibHandle)
    {
      /* Get the address of the custom cuda engine creation function if available
       * in the custom lib. */
        cudaEngineGetFcn = (NvDsInferCudaEngineGetFcn) dlsym(m_CustomLibHandle,
                "NvDsInferCudaEngineGet");
    }

    if (networkMode == NvDsInferNetworkMode_INT8)
    {
        /* Check if platform supports INT8 else use FP16 */
        if (builder->platformHasFastInt8())
        {
            if (!string_empty(initParams.int8CalibrationFilePath) &&
                    file_accessible(initParams.int8CalibrationFilePath))
            {
                /* Set INT8 mode and set the INT8 Calibrator */
                builder->setInt8Mode(true);
                builder->setInt8Calibrator(&pCalibrator);
                /* modelDataType should be FLOAT for INT8 */
                modelDataType = DataType::kFLOAT;
            }
            else if (cudaEngineGetFcn != nullptr)
            {
                printWarning("INT8 calibration file not specified/accessible. "
                        "INT8 calibration can be done through setDynamicRange "
                        "API in 'NvDsInferCreateNetwork' implementation");
            }
            else
            {
                printWarning("INT8 calibration file not specified. Trying FP16 mode.");
                networkMode = NvDsInferNetworkMode_FP16;
            }
        }
        else
        {
            printWarning("INT8 not supported by platform. Trying FP16 mode.");
            networkMode = NvDsInferNetworkMode_FP16;
        }
    }

    if (networkMode == NvDsInferNetworkMode_FP16)
    {
        /* Check if platform supports FP16 else use FP32 */
        if (builder->platformHasFastFp16())
        {
            builder->setHalf2Mode(true);
            modelDataType = DataType::kHALF;
        }
        else
        {
            printWarning("FP16 not supported by platform. Using FP32 mode.");
            networkMode = NvDsInferNetworkMode_FP32;
        }
    }

    if (networkMode == NvDsInferNetworkMode_FP32)
    {
        modelDataType = DataType::kFLOAT;
    }

    /* Set the maximum batch size */
    builder->setMaxBatchSize(m_MaxBatchSize);
    builder->setMaxWorkspaceSize(WORKSPACE_SIZE);

    /* Use DLA if specified. */
    if (initParams.useDLA)
    {
        builder->setDefaultDeviceType(DeviceType::kDLA);
        builder->setDLACore(initParams.dlaCore);
        builder->allowGPUFallback(true);
    }

    /* If the custom network creation function has been specified use that. */
    if (cudaEngineGetFcn)
    {
        nvinfer1::ICudaEngine *engine = nullptr;
        if (!cudaEngineGetFcn (builder.get(), &initParams, modelDataType, engine) ||
                engine == nullptr)
        {
            printError("Failed to create network using custom network creation"
                    " function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
        cudaEngine = engine;
        if (!string_empty(initParams.tltEncodedModelFilePath))
        {
            engineFileName << initParams.tltEncodedModelFilePath;
        }
        else
        {
            char *cwd = getcwd(NULL, 0);
            engineFileName << cwd << "/model";
            free(cwd);
        }
    }
    /* Check for caffe model files first. */
    else if (!string_empty(initParams.modelFilePath) &&
            !string_empty(initParams.protoFilePath))
    {
        if (!file_accessible(initParams.modelFilePath))
        {
            printError("Cannot access caffemodel file '%s'",
                    initParams.modelFilePath);
            return NVDSINFER_CONFIG_FAILED;
        }
        if (!file_accessible(initParams.protoFilePath))
        {
            printError("Cannot access prototxt file '%s'",
                    initParams.protoFilePath);
            return NVDSINFER_CONFIG_FAILED;
        }

        caffeParser = nvcaffeparser1::createCaffeParser();
        /* Check if the custom library provides a PluginFactory for Caffe parsing. */
        if (m_CustomLibHandle)
        {
            NvDsInferPluginFactoryCaffeGetFcn fcn =
                (NvDsInferPluginFactoryCaffeGetFcn) dlsym(m_CustomLibHandle,
                        "NvDsInferPluginFactoryCaffeGet");
            if (fcn)
            {
                NvDsInferPluginFactoryType type;
                if (!fcn(caffePluginFactory, type))
                {
                    printError("Could not get PluginFactory instance for "
                        "Caffe parsing from custom library");
                    return NVDSINFER_CUSTOM_LIB_FAILED;
                }
                /* Use the appropriate API to set the PluginFactory based on its
                 * type. */
                switch (type)
                {
                    case PLUGIN_FACTORY:
                        caffeParser->setPluginFactory(
                                caffePluginFactory.pluginFactory);
                        break;
                    case PLUGIN_FACTORY_EXT:
                        caffeParser->setPluginFactoryExt(
                                caffePluginFactory.pluginFactoryExt);
                        break;
                    case PLUGIN_FACTORY_V2:
                        caffeParser->setPluginFactoryV2(
                                caffePluginFactory.pluginFactoryV2);
                        break;
                    default:
                        printError("Invalid PluginFactory type returned by "
                            "custom library");
                        return NVDSINFER_CUSTOM_LIB_FAILED;
                }
            }
        }

        /* Parse the caffe model. */
        const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor =
                caffeParser->parse(initParams.protoFilePath,
                        initParams.modelFilePath, *network,
                        modelDataType);

        if (!blobNameToTensor)
        {
            printError("Failed while parsing network");
            return NVDSINFER_TENSORRT_ERROR;
        }

        for (unsigned int i = 0; i < initParams.numOutputLayers; i++)
        {
            char *layerName = initParams.outputLayerNames[i];
            /* Find and mark the coverage layer as output */
            ITensor *tensor = blobNameToTensor->find(layerName);
            if (!tensor)
            {
                printError("Could not find output layer '%s'", layerName);
                return NVDSINFER_CONFIG_FAILED;
            }
            network->markOutput(*tensor);
        }
        engineFileName << initParams.modelFilePath;
    }
    /* Check for UFF model next. */
    else if (!string_empty(initParams.uffFilePath))
    {
        if (!file_accessible(initParams.uffFilePath))
        {
            printError("Cannot access UFF file '%s'", initParams.uffFilePath);
            return NVDSINFER_CONFIG_FAILED;
        }

        //uffParser = nvuffparser::createUffParser();
        DimsCHW uffInputDims;
        nvuffparser::UffInputOrder uffInputOrder;

        /* UFF parsing needs the input layer name. */
        if (string_empty(initParams.uffInputBlobName))
        {
            printError("UFF input blob name not provided");
            return NVDSINFER_CONFIG_FAILED;

        }

        uffInputDims.c() = initParams.uffDimsCHW.c;
        uffInputDims.h() = initParams.uffDimsCHW.h;
        uffInputDims.w() = initParams.uffDimsCHW.w;

        switch (initParams.uffInputOrder)
        {
            case NvDsInferUffInputOrder_kNCHW:
                uffInputOrder = nvuffparser::UffInputOrder::kNCHW;
                break;
            case NvDsInferUffInputOrder_kNHWC:
                uffInputOrder = nvuffparser::UffInputOrder::kNHWC;
                break;
            case NvDsInferUffInputOrder_kNC:
                uffInputOrder = nvuffparser::UffInputOrder::kNC;
                break;
            default:
                printError("Unrecognized uff input order");
                return NVDSINFER_CONFIG_FAILED;
        }

        /* Register the input layer (name, dims and input order). */
        if (!uffParser->registerInput(initParams.uffInputBlobName,
                    uffInputDims, uffInputOrder))
        {
            printError("Failed to register input blob: %s DimsCHW:(%d,%d,%d) "
                "Order: %s", initParams.uffInputBlobName, initParams.uffDimsCHW.c,
                initParams.uffDimsCHW.h, initParams.uffDimsCHW.w,
                (initParams.uffInputOrder == NvDsInferUffInputOrder_kNHWC ?
                 "HWC" : "CHW"));
            return NVDSINFER_CONFIG_FAILED;

        }
        /* Register outputs. */
        for (unsigned int i = 0; i < initParams.numOutputLayers; i++) {
            uffParser->registerOutput(initParams.outputLayerNames[i]);
        }

        /* Check if the custom library provides a PluginFactory for UFF parsing. */
        if (m_CustomLibHandle)
        {
            NvDsInferPluginFactoryUffGetFcn fcn =
                (NvDsInferPluginFactoryUffGetFcn) dlsym(m_CustomLibHandle,
                        "NvDsInferPluginFactoryUffGet");
            if (fcn)
            {
                NvDsInferPluginFactoryType type;
                if (!fcn(uffPluginFactory, type))
                {
                    printError("Could not get PluginFactory instance for UFF"
                        " parsing from custom library");
                    return NVDSINFER_CUSTOM_LIB_FAILED;
                }
                /* Use the appropriate API to set the PluginFactory based on its
                 * type. */
                switch (type)
                {
                    case PLUGIN_FACTORY:
                        uffParser->setPluginFactory(
                                uffPluginFactory.pluginFactory);
                        break;
                    case PLUGIN_FACTORY_EXT:
                        uffParser->setPluginFactoryExt(
                                uffPluginFactory.pluginFactoryExt);
                        break;
                    default:
                        printError("Invalid PluginFactory type returned by "
                            "custom library");
                        return NVDSINFER_CUSTOM_LIB_FAILED;
                }
            }
        }

        if (!uffParser->parse(initParams.uffFilePath,
                    *network, modelDataType))
        {
            printError("Failed to parse UFF file: incorrect file or incorrect"
                " input/output blob names");
            return NVDSINFER_TENSORRT_ERROR;
        }
        engineFileName << initParams.uffFilePath;
    }
    else if (!string_empty(initParams.onnxFilePath))
    {
        if (!file_accessible(initParams.onnxFilePath))
        {
            printError("Cannot access ONNX file '%s'", initParams.onnxFilePath);
            return NVDSINFER_CONFIG_FAILED;
        }
        onnxParser = nvonnxparser::createParser(*network, m_Logger);

        if (!onnxParser->parseFromFile(initParams.onnxFilePath,
                    (int) ILogger::Severity::kWARNING))
        {
            printError("Failed to parse onnx file");
            return NVDSINFER_TENSORRT_ERROR;
        }
        engineFileName << initParams.onnxFilePath;
    }
    else
    {
        printError("No model files specified");
        return NVDSINFER_CONFIG_FAILED;
    }

    if (!cudaEngineGetFcn)
    {
        /* Build the engine */
        cudaEngine = builder->buildCudaEngine(*network);
    }
    if (cudaEngine == nullptr)
    {
        printError("Failed while building cuda engine for network");
        return NVDSINFER_TENSORRT_ERROR;
    }

    /* Serialize the network into a stream and return the stream pointer since
     * the cuda engine is valid only for the lifetime of the builder. */
    gieModelStream = cudaEngine->serialize();

    /* Optionally write the stream to a file which can used during next run. */
    engineFileName << "_b" << m_MaxBatchSize << "_";
    if (initParams.useDLA)
        engineFileName << "dla_";
    engineFileName << ((networkMode == NvDsInferNetworkMode_FP32) ? "fp32" :
                (networkMode == NvDsInferNetworkMode_FP16) ? "fp16" : "int8")
            << ".engine";
    printInfo("Storing the serialized cuda engine to file at %s",
            engineFileName.str().c_str());
    ofstream gieModelFileOut(engineFileName.str());
    gieModelFileOut.write((char *) gieModelStream->data(),
                          gieModelStream->size());

    cudaEngine.reset ();

    /* Destroy the plugin factory instances. */
    if (caffePluginFactory.pluginFactory)
    {
        NvDsInferPluginFactoryCaffeDestroyFcn fcn =
            (NvDsInferPluginFactoryCaffeDestroyFcn) dlsym(m_CustomLibHandle,
                    "NvDsInferPluginFactoryCaffeDestroy");
        if (fcn)
        {
            fcn(caffePluginFactory);
        }
    }
    if (uffPluginFactory.pluginFactory)
    {
        NvDsInferPluginFactoryUffDestroyFcn fcn =
            (NvDsInferPluginFactoryUffDestroyFcn) dlsym(m_CustomLibHandle,
                    "NvDsInferPluginFactoryUffDestroy");
        if (fcn)
        {
            fcn(uffPluginFactory);
        }
    }

    return NVDSINFER_SUCCESS;
}

/**
 * Clean up and free all resources
 */
NvDsInferContextImpl::~NvDsInferContextImpl()
{
    /* Set the cuda device to be used. */
    cudaError_t cudaReturn = cudaSetDevice(m_GpuID);
    if (cudaReturn != cudaSuccess)
    {
        printError("Failed to set cuda device %d (%s).", m_GpuID,
                cudaGetErrorName(cudaReturn));
        return;
    }

    unique_lock < std::mutex > lock (m_QueueMutex);

    /* Clean up other cuda resources. */
    if (m_PreProcessStream)
    {
        cudaStreamSynchronize(m_PreProcessStream);
        cudaStreamDestroy(m_PreProcessStream);
    }
    if (m_InferStream)
    {
        cudaStreamSynchronize(m_InferStream);
        cudaStreamDestroy(m_InferStream);
    }
    if (m_BufferCopyStream)
    {
        cudaStreamSynchronize(m_BufferCopyStream);
        cudaStreamDestroy(m_BufferCopyStream);
    }
    if (m_InputConsumedEvent)
        cudaEventDestroy (m_InputConsumedEvent);
    if (m_PreProcessCompleteEvent)
        cudaEventDestroy (m_PreProcessCompleteEvent);
    if (m_InferCompleteEvent)
        cudaEventDestroy (m_InferCompleteEvent);

    bool warn = false;

    for (auto & batch:m_Batches)
    {
        if (!batch.m_BuffersWithContext && !warn)
        {
            warn = true;
            printWarning ("Not all output batches released back to the context "
                    "before destroy. Memory associated with the outputs will "
                    "no longer be valid.");
        }
        if (batch.m_CopyCompleteEvent)
            cudaEventDestroy(batch.m_CopyCompleteEvent);
        for (size_t i = 0; i < batch.m_DeviceBuffers.size(); i++)
        {
            if (batch.m_DeviceBuffers[i] && !m_CudaEngine->bindingIsInput(i))
                cudaFree(batch.m_DeviceBuffers[i]);
        }
    }


    if (m_DBScanHandle)
        NvDsInferDBScanDestroy(m_DBScanHandle);

    if (m_InferExecutionContext)
        m_InferExecutionContext->destroy();

    if (m_CudaEngine)
        m_CudaEngine->destroy();

    if (m_InferRuntime)
        m_InferRuntime->destroy();

    if (m_CustomLibHandle)
    {
        /* Destroy the PluginFactory instance required during runtime cuda engine
         * deserialization. */
        if (m_RuntimePluginFactory)
        {
            NvDsInferPluginFactoryRuntimeDestroyFcn fcn =
                (NvDsInferPluginFactoryRuntimeDestroyFcn) dlsym(
                        m_CustomLibHandle, "NvDsInferPluginFactoryRuntimeDestroy");
            if (fcn)
            {
                fcn(m_RuntimePluginFactory);
            }
        }
        dlclose(m_CustomLibHandle);
    }

    if (m_MeanDataBuffer)
    {
        cudaFree(m_MeanDataBuffer);
    }

    for (auto & buffer:m_BindingBuffers)
    {
        if (buffer)
            cudaFree(buffer);
    }
}

/*
 * Destroy the context to release all resources.
 */
void
NvDsInferContextImpl::destroy()
{
    delete this;
}

/*
 * Factory function to create an NvDsInferContext instance and initialize it with
 * supplied parameters.
 */
NvDsInferStatus
createNvDsInferContext(NvDsInferContextHandle *handle,
        NvDsInferContextInitParams &initParams, void *userCtx,
        NvDsInferContextLoggingFunc logFunc)
{
    NvDsInferStatus status;
    NvDsInferContextImpl *ctx = new NvDsInferContextImpl();

    status = ctx->initialize(initParams, userCtx, logFunc);
    if (status == NVDSINFER_SUCCESS)
    {
        *handle = ctx;
    }
    else
    {
        static_cast<INvDsInferContext *>(ctx)->destroy();
    }
    return status;
}

/*
 * Reset the members inside the initParams structure to default values.
 */
void
NvDsInferContext_ResetInitParams (NvDsInferContextInitParams *initParams)
{
    if (initParams == nullptr)
    {
        fprintf(stderr, "Warning. NULL initParams passed to "
                "NvDsInferContext_ResetInitParams()\n");
        return;
    }

    memset(initParams, 0, sizeof (*initParams));

    initParams->networkMode = NvDsInferNetworkMode_FP32;
    initParams->networkInputFormat = NvDsInferFormat_Unknown;
    initParams->uffInputOrder = NvDsInferUffInputOrder_kNCHW;
    initParams->maxBatchSize = 1;
    initParams->networkScaleFactor = 1.0;
    initParams->networkType = NvDsInferNetworkType_Detector;
    initParams->outputBufferPoolSize = NVDSINFER_MIN_OUTPUT_BUFFERPOOL_SIZE;
}

const char *
NvDsInferContext_GetStatusName (NvDsInferStatus status)
{
#define CHECK_AND_RETURN_STRING(status_iter) \
    if (status == status_iter) return #status_iter

    CHECK_AND_RETURN_STRING(NVDSINFER_SUCCESS);
    CHECK_AND_RETURN_STRING(NVDSINFER_CONFIG_FAILED);
    CHECK_AND_RETURN_STRING(NVDSINFER_CUSTOM_LIB_FAILED);
    CHECK_AND_RETURN_STRING(NVDSINFER_INVALID_PARAMS);
    CHECK_AND_RETURN_STRING(NVDSINFER_OUTPUT_PARSING_FAILED);
    CHECK_AND_RETURN_STRING(NVDSINFER_CUDA_ERROR);
    CHECK_AND_RETURN_STRING(NVDSINFER_TENSORRT_ERROR);
    CHECK_AND_RETURN_STRING(NVDSINFER_UNKNOWN_ERROR);

    return nullptr;
#undef CHECK_AND_RETURN_STRING

}
