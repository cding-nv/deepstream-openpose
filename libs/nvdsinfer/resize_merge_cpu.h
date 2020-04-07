#ifndef RESIZE_MERGE_CPU_H
#define RESIZE_MERGE_CPU_H

#include <array>
#include <vector>

    template <typename T>
    void resizeAndMergeCpu(
        T* targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<T>& scaleInputToNetInputs = {1.f});

#endif // RESIZE_MERGE_CPU_H
