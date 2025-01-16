// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

#include <cub/device/device_merge_sort.cuh>
#include <AdePT/core/CublasWrappers.cuh>
#include <AdePT/core/ScoringCommons.hh>
#include <stdio.h>

// template <>
// cudaError_t cub::DeviceMergeSort::SortKeys(
//     void*,
//     std::size_t&,
//     GPUHit*,
//     unsigned int,
//     AsyncAdePT::CompareGPUHits,
//     cudaStream_t);

namespace cublas_wrappers {
template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
cudaError_t CublasSortKeys(void *d_temp_storage, std::size_t &temp_storage_bytes, KeyIteratorT d_keys,
                           OffsetT num_items, CompareOpT compare_op, cudaStream_t stream)
{
    printf("Cublas sort called\n");
    return cub::DeviceMergeSort::SortKeys(d_temp_storage, 
                                          temp_storage_bytes, 
                                          d_keys, 
                                          num_items,
                                          compare_op,
                                          stream);
}

// Explicit instantiations of these templates

// From HitScoring::HitScoring
template cudaError_t CublasSortKeys<GPUHit*, unsigned int, CompareGPUHits>
                                        (void*, 
                                        std::size_t &, 
                                        GPUHit*, 
                                        unsigned int,
                                        CompareGPUHits, 
                                        cudaStream_t);

} // namespace cublas_wrappers
