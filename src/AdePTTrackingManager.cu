// SPDX-FileCopyrightText: 2023 CERN
// SPDX-License-Identifier: Apache-2.0

// #include <AdePT/core/AdePTTransport.cuh>
// #include <AdePT/core/AsyncAdePTTransport.cuh>
#include <AdePT/integration/AdePTGeant4Integration.hh>

#ifndef ASYNC_MODE
#include <AdePT/core/AdePTTransport.cuh>
#else
#include <AdePT/core/AsyncAdePTTransport.cuh>
// #include <cub/device/device_merge_sort.cuh>
#endif

#ifndef ASYNC_MODE

// Explicit instantiation of the ShowerGPU<AdePTGeant4Integration> function
namespace adept_impl {
template void ShowerGPU<AdePTGeant4Integration>(AdePTGeant4Integration &, int, adeptint::TrackBuffer &, GPUstate &,
                                                HostScoring *, HostScoring *);
} // namespace adept_impl

#else

// namespace cublas_wrappers
// {

// template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
// cudaError_t CublasSortKeys(
//     void* d_temp_storage,
//     std::size_t& temp_storage_bytes,
//     KeyIteratorT d_keys,
//     OffsetT num_items,
//     CompareOpT compare_op,
//     cudaStream_t stream)
// {
//     printf("CublasSortKeys Called\n");   
// }
// }

// namespace async_adept_impl
// {
// void AsyncAdePTTransport::TransportLoop();
// }

#endif

