#ifndef NMS_CPU_H
#define NMS_CPU_H

#include <array>

template <typename T>
    void nmsCpu(
      T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize,
      const std::array<int, 4>& sourceSize);
      
#endif // NMS_CPU_H