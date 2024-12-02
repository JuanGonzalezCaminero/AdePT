// SPDX-FileCopyrightText: 2025 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef CUBLAS_WRAPPERS_CUH
#define CUBLAS_WRAPPERS_CUH

namespace cublas_wrappers {
  template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
  cudaError_t CublasSortKeys(
      void* d_temp_storage,
      std::size_t& temp_storage_bytes,
      KeyIteratorT d_keys,
      OffsetT num_items,
      CompareOpT compare_op,
      cudaStream_t stream = 0);
}

#endif