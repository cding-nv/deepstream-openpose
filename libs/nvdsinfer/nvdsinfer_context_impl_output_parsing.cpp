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
#include <iostream>
//#include <glib-2.0/glib.h>
#include <gst/gst.h>

#include "nvdsinfer_context_impl.h"
#include "nms_cpu.h"
#include "resize_merge_cpu.h"

static const bool ATHR_ENABLED = true;
static const float ATHR_THRESHOLD = 60.0;

using namespace std;

#define DIVIDE_AND_ROUND_UP(a, b) ((a + b - 1) / b)

/* Parse all object bounding boxes for the class `classIndex` in the frame
 * meeting the minimum threshold criteria.
 *
 * This parser function has been specifically written for the sample resnet10
 * model provided with the SDK. Other models will require this function to be
 * modified.
 */
bool
NvDsInferContextImpl::parseBoundingBox(
    vector < NvDsInferLayerInfo > const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    vector < NvDsInferObjectDetectionInfo > &objectList) {

    int outputCoverageLayerIndex = -1;
    int outputBBoxLayerIndex = -1;


    for (unsigned int i = 0; i < outputLayersInfo.size(); i++) {
        if (strstr(outputLayersInfo[i].layerName, "bbox") != nullptr) {
            outputBBoxLayerIndex = i;
        }
        if (strstr(outputLayersInfo[i].layerName, "cov") != nullptr) {
            outputCoverageLayerIndex = i;
        }
    }

    if (outputCoverageLayerIndex == -1) {
        printError("Could not find output coverage layer for parsing objects");
        return false;
    }
    if (outputBBoxLayerIndex == -1) {
        printError("Could not find output bbox layer for parsing objects");
        return false;
    }

    float *outputCoverageBuffer =
        (float *)outputLayersInfo[outputCoverageLayerIndex].buffer;
    float *outputBboxBuffer =
        (float *)outputLayersInfo[outputBBoxLayerIndex].buffer;

    NvDsInferDimsCHW outputCoverageDims;
    NvDsInferDimsCHW outputBBoxDims;

    getDimsCHWFromDims(outputCoverageDims,
                       outputLayersInfo[outputCoverageLayerIndex].dims);
    getDimsCHWFromDims(outputBBoxDims,
                       outputLayersInfo[outputBBoxLayerIndex].dims);

    unsigned int targetShape[2] = { outputCoverageDims.w, outputCoverageDims.h };
    float bboxNorm[2] = { 35.0, 35.0 };
    float gcCenters0[targetShape[0]];
    float gcCenters1[targetShape[1]];
    int gridSize = outputCoverageDims.w * outputCoverageDims.h;
    int strideX = DIVIDE_AND_ROUND_UP(networkInfo.width, outputBBoxDims.w);
    int strideY = DIVIDE_AND_ROUND_UP(networkInfo.height, outputBBoxDims.h);

    for (unsigned int i = 0; i < targetShape[0]; i++) {
        gcCenters0[i] = (float)(i * strideX + 0.5);
        gcCenters0[i] /= (float)bboxNorm[0];
    }
    for (unsigned int i = 0; i < targetShape[1]; i++) {
        gcCenters1[i] = (float)(i * strideY + 0.5);
        gcCenters1[i] /= (float)bboxNorm[1];
    }

    unsigned int numClasses =
        MIN(outputCoverageDims.c, detectionParams.numClassesConfigured);
    for (unsigned int classIndex = 0; classIndex < numClasses; classIndex++) {

        /* Pointers to memory regions containing the (x1,y1) and (x2,y2) coordinates
         * of rectangles in the output bounding box layer. */
        float *outputX1 = outputBboxBuffer
                          + classIndex * sizeof (float) * outputBBoxDims.h * outputBBoxDims.w;

        float *outputY1 = outputX1 + gridSize;
        float *outputX2 = outputY1 + gridSize;
        float *outputY2 = outputX2 + gridSize;

        /* Iterate through each point in the grid and check if the rectangle at that
         * point meets the minimum threshold criteria. */
        for (unsigned int h = 0; h < outputCoverageDims.h; h++) {
            for (unsigned int w = 0; w < outputCoverageDims.w; w++) {
                int i = w + h * outputCoverageDims.w;
                float confidence = outputCoverageBuffer[classIndex * gridSize + i];

                if (confidence < detectionParams.perClassThreshold[classIndex])
                    continue;

                int rectX1, rectY1, rectX2, rectY2;
                float rectX1Float, rectY1Float, rectX2Float, rectY2Float;

                /* Centering and normalization of the rectangle. */
                rectX1Float =
                    outputX1[w + h * outputCoverageDims.w] - gcCenters0[w];
                rectY1Float =
                    outputY1[w + h * outputCoverageDims.w] - gcCenters1[h];
                rectX2Float =
                    outputX2[w + h * outputCoverageDims.w] + gcCenters0[w];
                rectY2Float =
                    outputY2[w + h * outputCoverageDims.w] + gcCenters1[h];

                rectX1Float *= -bboxNorm[0];
                rectY1Float *= -bboxNorm[1];
                rectX2Float *= bboxNorm[0];
                rectY2Float *= bboxNorm[1];

                rectX1 = rectX1Float;
                rectY1 = rectY1Float;
                rectX2 = rectX2Float;
                rectY2 = rectY2Float;

                /* Clip parsed rectangles to frame bounds. */
                if (rectX1 >= (int)m_NetworkInfo.width)
                    rectX1 = m_NetworkInfo.width - 1;
                if (rectX2 >= (int)m_NetworkInfo.width)
                    rectX2 = m_NetworkInfo.width - 1;
                if (rectY1 >= (int)m_NetworkInfo.height)
                    rectY1 = m_NetworkInfo.height - 1;
                if (rectY2 >= (int)m_NetworkInfo.height)
                    rectY2 = m_NetworkInfo.height - 1;

                if (rectX1 < 0)
                    rectX1 = 0;
                if (rectX2 < 0)
                    rectX2 = 0;
                if (rectY1 < 0)
                    rectY1 = 0;
                if (rectY2 < 0)
                    rectY2 = 0;

                objectList.push_back({ classIndex, (unsigned int) rectX1,
                                       (unsigned int) rectY1, (unsigned int) (rectX2 - rectX1),
                                       (unsigned int) (rectY2 - rectY1), confidence});
            }
        }
    }
    return true;
}

/**
 * Cluster objects using OpenCV groupRectangles and fill the output structure.
 */
void
NvDsInferContextImpl::clusterAndFillDetectionOutputCV(NvDsInferDetectionOutput &output) {
    size_t totalObjects = 0;

    for (auto & list:m_PerClassCvRectList)
        list.clear();

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (auto & object:m_ObjectList) {
        m_PerClassCvRectList[object.classId].emplace_back(object.left,
                object.top, object.width, object.height);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object. Refer
         * to opencv documentation of groupRectangles for more
         * information about the tuning parameters for grouping. */
        if (m_PerClassDetectionParams[c].groupThreshold > 0)
            cv::groupRectangles(m_PerClassCvRectList[c],
                                m_PerClassDetectionParams[c].groupThreshold,
                                m_PerClassDetectionParams[c].eps);
        totalObjects += m_PerClassCvRectList[c].size();
    }

    output.objects = new NvDsInferObject[totalObjects];
    output.numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (auto & rect:m_PerClassCvRectList[c]) {
            NvDsInferObject &object = output.objects[output.numObjects];
            object.left = rect.x;
            object.top = rect.y;
            object.width = rect.width;
            object.height = rect.height;
            object.classIndex = c;
            object.label = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            output.numObjects++;
        }
    }
}

/**
 * Cluster objects using DBSCAN and fill the output structure.
 */
void
NvDsInferContextImpl::clusterAndFillDetectionOutputDBSCAN(NvDsInferDetectionOutput &output) {
    size_t totalObjects = 0;
    NvDsInferDBScanClusteringParams clusteringParams;
    clusteringParams.enableATHRFilter = ATHR_ENABLED;
    clusteringParams.thresholdATHR = ATHR_THRESHOLD;
    vector<size_t> numObjectsList(m_NumDetectedClasses);

    for (auto & list:m_PerClassObjectList)
        list.clear();

    /* The above functions will add all objects in the m_ObjectList vector.
     * Need to seperate them per class for grouping. */
    for (auto & object:m_ObjectList) {
        m_PerClassObjectList[object.classId].emplace_back(object);
    }

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        NvDsInferObjectDetectionInfo *objArray = m_PerClassObjectList[c].data();
        size_t numObjects = m_PerClassObjectList[c].size();

        clusteringParams.eps = m_PerClassDetectionParams[c].eps;
        clusteringParams.minBoxes = m_PerClassDetectionParams[c].minBoxes;

        /* Cluster together rectangles with similar locations and sizes
         * since these rectangles might represent the same object using
         * DBSCAN. */
        if (m_PerClassDetectionParams[c].minBoxes > 0)
            NvDsInferDBScanCluster(m_DBScanHandle, &clusteringParams,
                                   objArray, &numObjects);
        totalObjects += numObjects;
        numObjectsList[c] = numObjects;
    }

    output.objects = new NvDsInferObject[totalObjects];
    output.numObjects = 0;

    for (unsigned int c = 0; c < m_NumDetectedClasses; c++) {
        /* Add coordinates and class ID and the label of all objects
         * detected in the frame to the frame output. */
        for (size_t i = 0; i < numObjectsList[c]; i++) {
            NvDsInferObject &object = output.objects[output.numObjects];
            object.left = m_PerClassObjectList[c][i].left;
            object.top = m_PerClassObjectList[c][i].top;
            object.width = m_PerClassObjectList[c][i].width;
            object.height = m_PerClassObjectList[c][i].height;
            object.classIndex = c;
            object.label = nullptr;
            if (c < m_Labels.size() && m_Labels[c].size() > 0)
                object.label = strdup(m_Labels[c][0].c_str());
            output.numObjects++;
        }
    }
}

bool
NvDsInferContextImpl::parseAttributesFromSoftmaxLayers(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo  const &networkInfo,
    float classifierThreshold,
    std::vector<NvDsInferAttribute> &attrList,
    std::string &attrString) {
    /* Get the number of attributes supported by the classifier. */
    unsigned int numAttributes = m_OutputLayerInfo.size();

    /* Iterate through all the output coverage layers of the classifier.
    */
    for (unsigned int l = 0; l < numAttributes; l++) {
        /* outputCoverageBuffer for classifiers is usually a softmax layer.
         * The layer is an array of probabilities of the object belonging
         * to each class with each probability being in the range [0,1] and
         * sum all probabilities will be 1.
         */
        NvDsInferDimsCHW dims;

        getDimsCHWFromDims(dims, m_OutputLayerInfo[l].dims);
        unsigned int numClasses = dims.c;
        float *outputCoverageBuffer =
            (float *)m_OutputLayerInfo[l].buffer;
        float maxProbability = 0;
        bool attrFound = false;
        NvDsInferAttribute attr;

        /* Iterate through all the probabilities that the object belongs to
         * each class. Find the maximum probability and the corresponding class
         * which meets the minimum threshold. */
        for (unsigned int c = 0; c < numClasses; c++) {
            float probability = outputCoverageBuffer[c];
            if (probability > m_ClassifierThreshold
                    && probability > maxProbability) {
                maxProbability = probability;
                attrFound = true;
                attr.attributeIndex = l;
                attr.attributeValue = c;
                attr.attributeConfidence = probability;
            }
        }
        if (attrFound) {
            if (m_Labels.size() > attr.attributeIndex &&
                    attr.attributeValue < m_Labels[attr.attributeIndex].size())
                attr.attributeLabel =
                    m_Labels[attr.attributeIndex][attr.attributeValue].c_str();
            else
                attr.attributeLabel = nullptr;
            attrList.push_back(attr);
            if (attr.attributeLabel)
                attrString.append(attr.attributeLabel).append(" ");
        }
    }

    return true;
}

NvDsInferStatus
NvDsInferContextImpl::fillDetectionOutput(NvDsInferDetectionOutput &output) {
    /* Clear the object lists. */
    m_ObjectList.clear();

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomBBoxParseFunc) {
        if (!m_CustomBBoxParseFunc(m_OutputLayerInfo, m_NetworkInfo,
                                   m_DetectionParams, m_ObjectList)) {
            printError("Failed to parse bboxes using custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        if (!parseBoundingBox(m_OutputLayerInfo, m_NetworkInfo,
                              m_DetectionParams, m_ObjectList)) {
            printError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    if (m_UseDBScan)
        clusterAndFillDetectionOutputDBSCAN(output);
    else
        clusterAndFillDetectionOutputCV(output);

    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::fillClassificationOutput(NvDsInferClassificationOutput &output) {
    string attrString;
    vector<NvDsInferAttribute> attributes;

    /* Call custom parsing function if specified otherwise use the one
     * written along with this implementation. */
    if (m_CustomClassifierParseFunc) {
        if (!m_CustomClassifierParseFunc(m_OutputLayerInfo, m_NetworkInfo,
                                         m_ClassifierThreshold, attributes, attrString)) {
            printError("Failed to parse classification attributes using "
                       "custom parse function");
            return NVDSINFER_CUSTOM_LIB_FAILED;
        }
    } else {
        if (!parseAttributesFromSoftmaxLayers(m_OutputLayerInfo, m_NetworkInfo,
                                              m_ClassifierThreshold, attributes, attrString)) {
            printError("Failed to parse bboxes");
            return NVDSINFER_OUTPUT_PARSING_FAILED;
        }
    }

    /* Fill the output structure with the parsed attributes. */
    output.label = strdup(attrString.c_str());
    output.numAttributes = attributes.size();
    output.attributes = new NvDsInferAttribute[output.numAttributes];
    for (size_t i = 0; i < output.numAttributes; i++) {
        output.attributes[i].attributeIndex = attributes[i].attributeIndex;
        output.attributes[i].attributeValue = attributes[i].attributeValue;
        output.attributes[i].attributeConfidence = attributes[i].attributeConfidence;
        output.attributes[i].attributeLabel = attributes[i].attributeLabel;
    }
    return NVDSINFER_SUCCESS;
}

NvDsInferStatus
NvDsInferContextImpl::fillSegmentationOutput(NvDsInferSegmentationOutput &output) {
    NvDsInferDimsCHW outputDimsCHW;
    getDimsCHWFromDims(outputDimsCHW, m_OutputLayerInfo[0].dims);

    //$6 = {numDims = 3, d = {57, 46, 62, 127, 2918418508, 127, 1443693648, 85}, numElements = 162564}

    const int SCALE = 8;
    output.width = outputDimsCHW.w * SCALE;    //62
    output.height = outputDimsCHW.h * SCALE;   //46
    output.classes = outputDimsCHW.c;          //57

    output.class_map = new int [output.width * output.height];
    output.class_probability_map = (float *) m_OutputLayerInfo[0].buffer;

    int out[46][62];
    for (int i = 0; i < 46; i++) {
        for (int j = 0; j < 62; j++) {
            out[i][j] = 1;
        }
    }
    for (int k = 0; k < 18; k++) {
        int below = 0;
        int x = 0, y = 0;
        float confidence = 0.0;

        for (int i = 0; i < 46; i++) {
            for (int j = 0; j < 62; j++) {
                if (output.class_probability_map[k*46*62 + i * 62 + j] > confidence) {
                    confidence = output.class_probability_map[k*46*62 + i * 62 + j];
                    x = j;
                    y = i;
                }

                if (output.class_probability_map[k*46*62 + i * 62 + j] < 0) {
                    below++;
                }

            }
        }
        out[y][x] = 0;

        //printf("k=%d,y/x=(%d,%d) below 0 = %d\n", k, y, x, below);
    }

    for (int i = 0; i < 46; i++) {
        for (int j = 0; j < 62; j++) {
            printf("%d", out[i][j]);
            for (int y = 0; y < SCALE; y++) {
                for (int x = 0; x < SCALE; x++) {
                    output.class_map[(i * SCALE + y) * output.width + j * SCALE + x] = 6 - out[i][j];
                }
            }
        }
        printf("\n");
    }

#if 1
    // Reszie and merge
    float* resize_target_ptr = (float*)malloc(sizeof(float) * 368*496*57);
    std::vector<const float*> resize_source_ptr = {output.class_probability_map};
    std::array<int, 4> resize_target_size = {1, 57, 368, 496};
    std::vector<std::array<int, 4>> resize_source_size = {{1, 57, 46, 62}};
    std::vector<float> scale_input_to_net_inputs = {1.0};

    resizeAndMergeCpu(resize_target_ptr, resize_source_ptr, resize_target_size, resize_source_size, scale_input_to_net_inputs);

    // nms
    float* nms_target_ptr = (float*)malloc(sizeof(float) * 18 * 128 * 3);
    int *  kernel_ptr = (int*)malloc(sizeof(int) * 368 * 496 * 57);
    float* nms_source_ptr = resize_target_ptr;
    float threshold = 0.05f;
    int outputChannels = 18;
    int POSE_MAX_PEOPLE = 127+1;
    int x_y_sorce = 3;

    std::array<int, 4> nms_target_size = {1, outputChannels, POSE_MAX_PEOPLE, x_y_sorce};
    std::array<int, 4> nms_source_size = {1, 57, 368, 496};

    nmsCpu(nms_target_ptr, kernel_ptr, nms_source_ptr, threshold, nms_target_size, nms_source_size);

    for (int i=0; i < outputChannels*POSE_MAX_PEOPLE / 3; i++) {
        if (nms_target_ptr[i*3+2] > 0.1)
            printf("%f %f %f\n", nms_target_ptr[i*3], nms_target_ptr[i*3+1], nms_target_ptr[i*3+2]);
    }
#endif

    output.classes = 1;


#if 0
    for (unsigned int y = 0; y < output.height; y++) {
        for (unsigned int x = 0; x < output.width; x++) {
            float max_prob = -1;
            int &cls = output.class_map[y * output.width + x] = -1;
            for (unsigned int c = 0; c < output.classes; c++) {
                float prob = output.class_probability_map[c * output.width * output.height + y * output.width + x];
                if (prob > max_prob && prob > m_SegmentationThreshold) {
                    cls = c;
                    max_prob = prob;
                }
            }
        }
    }
#endif

    return NVDSINFER_SUCCESS;
}

void
NvDsInferContextImpl::releaseFrameOutput(NvDsInferFrameOutput &frameOutput) {
    switch (m_NetworkType) {
    case NvDsInferNetworkType_Detector:
        for (unsigned int j = 0; j < frameOutput.detectionOutput.numObjects; j++) {
            free(frameOutput.detectionOutput.objects[j].label);
        }
        delete[] frameOutput.detectionOutput.objects;
        break;
    case NvDsInferNetworkType_Classifier:
        free(frameOutput.classificationOutput.label);
        delete[] frameOutput.classificationOutput.attributes;
        break;
    case NvDsInferNetworkType_Segmentation:
        delete[] frameOutput.segmentationOutput.class_map;
        break;
    default:
        break;
    }
}
