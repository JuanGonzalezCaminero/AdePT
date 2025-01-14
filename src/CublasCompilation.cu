// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <cub/device/device_merge_sort.cuh>
// #include <AdePT/core/ScoringCommons.hh>
#include <AdePT/core/PerEventScoringImpl.cuh>

// template <>
// cudaError_t cub::DeviceMergeSort::SortKeys(
//     void*,
//     std::size_t&,
//     GPUHit*,
//     unsigned int,
//     AsyncAdePT::CompareGPUHits,
//     cudaStream_t);

template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
cudaError_t AsyncAdePT::CublasSortKeys(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyIteratorT d_keys,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream)
{
    printf("CublasSortKeys Called\n");   
}