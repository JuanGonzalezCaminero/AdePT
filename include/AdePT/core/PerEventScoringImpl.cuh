// SPDX-FileCopyrightText: 2024 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef PER_EVENT_SCORING_CUH
#define PER_EVENT_SCORING_CUH

#include <AdePT/core/PerEventScoringStruct.cuh>
#include <AdePT/base/ResourceManagement.cuh>
#include <AdePT/core/AdePTScoringTemplate.cuh>
#include <AdePT/core/ScoringCommons.hh>
#include <AdePT/copcore/Global.h>

#include <VecGeom/navigation/NavigationState.h>

// namespace global_std = ::std;

// #include <thrust/sort.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <array>
#include <chrono>
#include <thread>

// namespace std = ::std;
// #ifdef __CUDA_ARCH__
// #include <cub/device/device_merge_sort.cuh>
// #endif

// namespace cub
// {
//   struct DeviceMergeSort;
// }

// Forward declarations of CUDA functions
// cub::DeviceMergeSort::SortKeys(nullptr, 
//                                 fGPUSortAuxMemorySize, 
//                                 fGPUHitBuffer_dev.get(), 
//                                 fHitCapacity, 
//                                 CompareGPUHits{});

namespace AsyncAdePT {

template <typename KeyIteratorT, typename OffsetT, typename CompareOpT>
cudaError_t CublasSortKeys(
    void* d_temp_storage,
    std::size_t& temp_storage_bytes,
    KeyIteratorT d_keys,
    OffsetT num_items,
    CompareOpT compare_op,
    cudaStream_t stream = 0);

// cudaError_t CublasSortKeys(void*, &std::size_t, fGPUHitBuffer_dev.get(), unsigned int, CompareGPUHits{});

// Comparison for sorting tracks into events on device:
struct CompareGPUHits {
  __device__ bool operator()(const GPUHit &lhs, const GPUHit &rhs) const { return lhs.fEventId < rhs.fEventId; }
};

/// Struct holding GPU hits to be used both on host and device.
struct HitScoringBuffer {
  GPUHit *hitBuffer_dev     = nullptr;
  unsigned int fSlotCounter = 0;
  unsigned int fNSlot       = 0;

  __device__ GPUHit &GetNextSlot()
  {
    const auto slotIndex = atomicAdd(&fSlotCounter, 1);
    if (slotIndex >= fNSlot) {
        printf("Trying to score hit #%d with only %d slots\n", slotIndex, fNSlot);
        COPCORE_EXCEPTION("Out of slots in HitScoringBuffer::NextSlot");
    }
    return hitBuffer_dev[slotIndex];
  }
};

__device__ HitScoringBuffer gHitScoringBuffer_dev;

struct BufferHandle {
  HitScoringBuffer hitScoringInfo;
  GPUHit *hostBuffer;
  enum class State { Free, OnDevice, OnDeviceNeedTransferToHost, TransferToHost, NeedHostProcessing };
  std::atomic<State> state;
};

class HitScoring {
  unique_ptr_cuda<GPUHit> fGPUHitBuffer_dev;
  unique_ptr_cuda<GPUHit, CudaHostDeleter<GPUHit>> fGPUHitBuffer_host;

  std::array<BufferHandle, 2> fBuffers;

  void *fHitScoringBuffer_deviceAddress = nullptr;
  unsigned int fHitCapacity;
  unsigned short fActiveBuffer = 0;
  unique_ptr_cuda<std::byte> fGPUSortAuxMemory;
  std::size_t fGPUSortAuxMemorySize;

  std::vector<std::deque<std::shared_ptr<const std::vector<GPUHit>>>> fHitQueues;
  mutable std::shared_mutex fProcessingHitsMutex;

  void ProcessBuffer(BufferHandle &handle)
  {
    // We are assuming that the caller holds a lock on fProcessingHitsMutex.
    if (handle.state == BufferHandle::State::NeedHostProcessing) {
        auto hitVector = std::make_shared<std::vector<GPUHit>>();
        hitVector->assign(handle.hostBuffer, handle.hostBuffer + handle.hitScoringInfo.fSlotCounter);
        handle.hitScoringInfo.fSlotCounter = 0;
        handle.state                       = BufferHandle::State::Free;

        for (auto &hitQueue : fHitQueues) {
        hitQueue.push_back(hitVector);
        }
    }
  }

public:
  // HitScoring(unsigned int hitCapacity, unsigned int nThread);
  HitScoring(unsigned int hitCapacity, unsigned int nThread) : fHitCapacity{hitCapacity}, fHitQueues(nThread)
  {
    // We use a single allocation for both buffers:
    GPUHit *gpuHits = nullptr;
    COPCORE_CUDA_CHECK(cudaMallocHost(&gpuHits, sizeof(GPUHit) * 2 * fHitCapacity));
    fGPUHitBuffer_host.reset(gpuHits);

    auto result = cudaMalloc(&gpuHits, sizeof(GPUHit) * 2 * fHitCapacity);
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit buffer."};
    fGPUHitBuffer_dev.reset(gpuHits);

    // Init buffers for on-device sorting of hits:
    // Determine device storage requirements for on-device sorting.
    // TODO: Enable sorting
    // result = cub::DeviceMergeSort::SortKeys(nullptr, fGPUSortAuxMemorySize, fGPUHitBuffer_dev.get(), fHitCapacity,
    //                                         CompareGPUHits{});
    result = CublasSortKeys(nullptr, fGPUSortAuxMemorySize, fGPUHitBuffer_dev.get(), fHitCapacity,
                                             CompareGPUHits{});
    // result = cudaSuccess;
    if (result != cudaSuccess) throw std::invalid_argument{"No space for hit sorting on device."};

    std::byte *gpuSortingMem;
    result = cudaMalloc(&gpuSortingMem, fGPUSortAuxMemorySize);
    // result = cudaSuccess;
    if (result != cudaSuccess) throw std::invalid_argument{"No space to allocate hit sorting buffer."};
    fGPUSortAuxMemory.reset(gpuSortingMem);

    // Store buffer data in structs
    fBuffers[0].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get(), 0, fHitCapacity};
    fBuffers[0].hostBuffer     = fGPUHitBuffer_host.get();
    fBuffers[0].state          = BufferHandle::State::OnDevice;
    fBuffers[1].hitScoringInfo = HitScoringBuffer{fGPUHitBuffer_dev.get() + fHitCapacity, 0, fHitCapacity};
    fBuffers[1].hostBuffer     = fGPUHitBuffer_host.get() + fHitCapacity;
    fBuffers[1].state          = BufferHandle::State::Free;

    COPCORE_CUDA_CHECK(cudaGetSymbolAddress(&fHitScoringBuffer_deviceAddress, gHitScoringBuffer_dev));
    assert(fHitScoringBuffer_deviceAddress != nullptr);
    COPCORE_CUDA_CHECK(cudaMemcpy(fHitScoringBuffer_deviceAddress, &fBuffers[0].hitScoringInfo, sizeof(HitScoringBuffer),
                                  cudaMemcpyHostToDevice));
  }

  unsigned int HitCapacity() const { return fHitCapacity; }
  
  void SwapDeviceBuffers(cudaStream_t cudaStream)
  {
     // Ensure that host side has been processed:
    auto &currentBuffer = fBuffers[fActiveBuffer];
    if (currentBuffer.state != BufferHandle::State::OnDevice)
        throw std::logic_error(__FILE__ + std::to_string(__LINE__) + ": On-device buffer in wrong state");

    // Get new buffer info from device:
    auto &currentHitInfo = currentBuffer.hitScoringInfo;
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(&currentHitInfo, fHitScoringBuffer_deviceAddress, sizeof(HitScoringBuffer),
                                        cudaMemcpyDefault, cudaStream));

    // Execute the swap:
    fActiveBuffer          = (fActiveBuffer + 1) % fBuffers.size();
    auto &nextDeviceBuffer = fBuffers[fActiveBuffer];
    while (nextDeviceBuffer.state != BufferHandle::State::Free) {
        std::cerr << __func__ << " Warning: Another thread should have processed the hits.\n";
    }
    assert(nextDeviceBuffer.state == BufferHandle::State::Free && nextDeviceBuffer.hitScoringInfo.fSlotCounter == 0);

    nextDeviceBuffer.state = BufferHandle::State::OnDevice;
    COPCORE_CUDA_CHECK(cudaMemcpyAsync(fHitScoringBuffer_deviceAddress, &nextDeviceBuffer.hitScoringInfo,
                                        sizeof(HitScoringBuffer), cudaMemcpyDefault, cudaStream));
    COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
    currentBuffer.state = BufferHandle::State::OnDeviceNeedTransferToHost;
    }
  
  bool ProcessHits()
  {
    std::unique_lock lock{fProcessingHitsMutex, std::defer_lock};
    bool haveNewHits = false;

    while (std::any_of(fBuffers.begin(), fBuffers.end(),
                        [](auto &buffer) { return buffer.state >= BufferHandle::State::TransferToHost; })) {
        for (auto &handle : fBuffers) {
        if (handle.state == BufferHandle::State::NeedHostProcessing) {
            if (!lock) lock.lock();
            haveNewHits = true;
            ProcessBuffer(handle);
        }
        }
    }

    return haveNewHits;
    }
  
  bool ReadyToSwapBuffers() const
  {
    return std::any_of(fBuffers.begin(), fBuffers.end(),
                       [](const auto &handle) { return handle.state == BufferHandle::State::Free; });
  }

  // void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy);
  void TransferHitsToHost(cudaStream_t cudaStreamForHitCopy)
  {
    for (auto &buffer : fBuffers) {
      if (buffer.state != BufferHandle::State::OnDeviceNeedTransferToHost) continue;

      buffer.state = BufferHandle::State::TransferToHost;
      assert(buffer.hitScoringInfo.fSlotCounter < fHitCapacity);

      auto bufferBegin = buffer.hitScoringInfo.hitBuffer_dev;

      // TODO: Sorting is disabled for now
      // cub::DeviceMergeSort::SortKeys(fGPUSortAuxMemory.get(), fGPUSortAuxMemorySize, bufferBegin,
      //                               buffer.hitScoringInfo.fSlotCounter, CompareGPUHits{}, cudaStreamForHitCopy);

      COPCORE_CUDA_CHECK(cudaMemcpyAsync(buffer.hostBuffer, bufferBegin,
                                        sizeof(GPUHit) * buffer.hitScoringInfo.fSlotCounter, cudaMemcpyDefault,
                                        cudaStreamForHitCopy));
      COPCORE_CUDA_CHECK(cudaLaunchHostFunc(
          cudaStreamForHitCopy,
          [](void *arg) { static_cast<BufferHandle *>(arg)->state = BufferHandle::State::NeedHostProcessing; }, &buffer));
    }
  }
  
  std::shared_ptr<const std::vector<GPUHit>> GetNextHitsVector(unsigned int threadId)
  {
    assert(threadId < fHitQueues.size());
    std::shared_lock lock{fProcessingHitsMutex};

    if (fHitQueues[threadId].empty())
      return nullptr;
    else {
      auto ret = fHitQueues[threadId].front();
      fHitQueues[threadId].pop_front();
      return ret;
    }
  }
};

// Implement Cuda-dependent functionality from PerEventScoring

void PerEventScoring::ClearGPU(cudaStream_t cudaStream)
{
  COPCORE_CUDA_CHECK(cudaMemsetAsync(fScoring_dev, 0, sizeof(GlobalCounters), cudaStream));
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
}

void PerEventScoring::CopyToHost(cudaStream_t cudaStream)
{
  const auto oldPointer = fScoring_dev;
  COPCORE_CUDA_CHECK(
      cudaMemcpyAsync(&fGlobalCounters, fScoring_dev, sizeof(GlobalCounters), cudaMemcpyDeviceToHost, cudaStream));
  COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
  assert(oldPointer == fScoring_dev);
  (void)oldPointer;
}

// struct PerEventScoring {
//   GlobalCounters fGlobalCounters;
//   PerEventScoring *const fScoring_dev;

//   PerEventScoring(PerEventScoring *gpuScoring) : fScoring_dev{gpuScoring} { ClearGPU(); }
//   PerEventScoring(PerEventScoring &&other) = default;
//   ~PerEventScoring()                       = default;

//   /// @brief Copy hits to host for a single event
//   void CopyToHost(cudaStream_t cudaStream = 0)
//   {
//     const auto oldPointer = fScoring_dev;
//     COPCORE_CUDA_CHECK(
//         cudaMemcpyAsync(&fGlobalCounters, fScoring_dev, sizeof(GlobalCounters), cudaMemcpyDeviceToHost, cudaStream));
//     COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
//     assert(oldPointer == fScoring_dev);
//     (void)oldPointer;
//   }

//   /// @brief Clear hits on device to reuse for next event
//   void ClearGPU(cudaStream_t cudaStream = 0)
//   {
//     COPCORE_CUDA_CHECK(cudaMemsetAsync(fScoring_dev, 0, sizeof(GlobalCounters), cudaStream));
//     COPCORE_CUDA_CHECK(cudaStreamSynchronize(cudaStream));
//   }

//   /// @brief Print scoring info
//   void Print() { fGlobalCounters.Print(); };
// };

} // namespace AsyncAdePT

namespace adept_scoring {

/// @brief Utility function to copy a 3D vector, used for filling the Step Points
__device__ __forceinline__ void Copy3DVector(vecgeom::Vector3D<Precision> const *source,
                                             vecgeom::Vector3D<Precision> *destination)
{
  destination->x() = source->x();
  destination->y() = source->y();
  destination->z() = source->z();
}

/// @brief Record a hit
template <>
__device__ void RecordHit(AsyncAdePT::PerEventScoring * /*scoring*/, int aParentID, char aParticleType,
                          double aStepLength, double aTotalEnergyDeposit, vecgeom::NavigationState const *aPreState,
                          vecgeom::Vector3D<Precision> const *aPrePosition,
                          vecgeom::Vector3D<Precision> const *aPreMomentumDirection,
                          vecgeom::Vector3D<Precision> const * /*aPrePolarization*/, double aPreEKin, double aPreCharge,
                          vecgeom::NavigationState const *aPostState, vecgeom::Vector3D<Precision> const *aPostPosition,
                          vecgeom::Vector3D<Precision> const *aPostMomentumDirection,
                          vecgeom::Vector3D<Precision> const * /*aPostPolarization*/, double aPostEKin,
                          double aPostCharge, unsigned int eventID, short threadID)
{
  // Acquire a hit slot
  GPUHit &aGPUHit  = AsyncAdePT::gHitScoringBuffer_dev.GetNextSlot();
  aGPUHit.fEventId = eventID;
  aGPUHit.threadId = threadID;

  // Fill the required data
  aGPUHit.fParentID           = aParentID;
  aGPUHit.fParticleType       = aParticleType;
  aGPUHit.fStepLength         = aStepLength;
  aGPUHit.fTotalEnergyDeposit = aTotalEnergyDeposit;
  // Pre step point
  aGPUHit.fPreStepPoint.fNavigationState = *aPreState;
  Copy3DVector(aPrePosition, &(aGPUHit.fPreStepPoint.fPosition));
  Copy3DVector(aPreMomentumDirection, &(aGPUHit.fPreStepPoint.fMomentumDirection));
  // Copy3DVector(aPrePolarization, aGPUHit.fPreStepPoint.fPolarization);
  aGPUHit.fPreStepPoint.fEKin   = aPreEKin;
  aGPUHit.fPreStepPoint.fCharge = aPreCharge;
  // Post step point
  aGPUHit.fPostStepPoint.fNavigationState = *aPostState;
  Copy3DVector(aPostPosition, &(aGPUHit.fPostStepPoint.fPosition));
  Copy3DVector(aPostMomentumDirection, &(aGPUHit.fPostStepPoint.fMomentumDirection));
  // Copy3DVector(aPostPolarization, aGPUHit.fPostStepPoint.fPolarization);
  aGPUHit.fPostStepPoint.fEKin   = aPostEKin;
  aGPUHit.fPostStepPoint.fCharge = aPostCharge;
}

/// @brief Account for the number of produced secondaries
/// @details Atomically increase the number of produced secondaries.
template <>
__device__ void AccountProduced(AsyncAdePT::PerEventScoring *scoring, int num_ele, int num_pos, int num_gam)
{
  atomicAdd(&scoring->fGlobalCounters.numElectrons, num_ele);
  atomicAdd(&scoring->fGlobalCounters.numPositrons, num_pos);
  atomicAdd(&scoring->fGlobalCounters.numGammas, num_gam);
}

// template <>
// inline void EndOfTransport(AsyncAdePT::PerEventScoring &scoring, AsyncAdePT::PerEventScoring *, cudaStream_t *, IntegrationLayer *)
// {
//   scoring.CopyToHost();
//   scoring.ClearGPU();

//   // TODO: this needs to be done by the caller, or the net energy array moved to the scoring data
//   // fGPUNetEnergy[threadId] = 0.;

//   // TODO: This isn't critical, if needed we need to add a debug level parameter to the interface
//   // if (fDebugLevel >= 2) {
//   //   G4cout << "\n\tScoring for event " << eventId << G4endl;
//   //   scoring.Print();
//   // }
// }

} // namespace adept_scoring

#endif
