// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef ASYNC_ADEPT_TRANSPORT_STRUCT_CUH
#define ASYNC_ADEPT_TRANSPORT_STRUCT_CUH

#include <AdePT/core/CommonStruct.h>
#include <AdePT/core/AsyncAdePTTransportStruct.hh>
// #include <AdePT/core/AsyncAdePTTransport.hh>
#include <AdePT/core/PerEventScoringImpl.cuh>
#include <AdePT/core/Track.cuh>
#include <AdePT/magneticfield/GeneralMagneticField.h>
#include <AdePT/magneticfield/UniformMagneticField.h>

#include <AdePT/base/SlotManager.cuh>
#include <AdePT/base/ResourceManagement.cuh>

#include <G4HepEmData.hh>
#include <G4HepEmParameters.hh>

#ifdef USE_SPLIT_KERNELS
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmGammaTrack.hh>
#endif

#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

namespace AsyncAdePT {

// A bundle of pointers to generate particles of an implicit type.
struct ParticleGenerator {
  Track *fTracks;
  SoATrack *fSoATracks;
  SlotManager *fSlotManager;
  SlotManager *fSlotManagerLeaks;
  adept::MParray *fActiveQueue;

public:
  __host__ __device__ ParticleGenerator(Track *tracks, SoATrack *soaTracks, SlotManager *slotManager,
                                        SlotManager *slotManagerLeaks, adept::MParray *activeQueue)
      : fTracks(tracks), fSoATracks(soaTracks), fSlotManager(slotManager), fSlotManagerLeaks(slotManagerLeaks),
        fActiveQueue(activeQueue)
  {
  }

  /// Obtain a slot for a track, but don't enqueue.
  __device__ auto NextSlot() { return fSlotManager->NextSlot(); }

  __device__ auto NextLeakSlot() { return fSlotManagerLeaks->NextSlot(); }

  /// Construct a track at the given location, forwarding all arguments to the constructor.
  template <typename... Ts>
  __device__ Track &InitTrack(SlotManager::value_type slot, Ts &&...args)
  {
    // Initialize the values in the SoA storage
    fSoATracks->InitTrack(slot, std::forward<Ts>(args)...);
    // Init the main track
    return *new (fTracks + slot) Track{std::forward<Ts>(args)...};
  }

  /// Obtain a slot and construct a track, forwarding args to the track constructor.
  template <typename... Ts>
  __device__ Track &NextTrack(Ts &&...args)
  {
    const auto slot = NextSlot();
    fActiveQueue->push_back(slot);
    // Initialize the values in the SoA storage
    fSoATracks->InitTrack(slot, std::forward<Ts>(args)...);
    // Init the main track
    auto &track = InitTrack(slot, std::forward<Ts>(args)...);
    // track.currentSlot = slot;
    return track;
  }

  void SetActiveQueue(adept::MParray *queue) { fActiveQueue = queue; }
};

struct LeakedTracks {
  Track *fLeaks;
  SoATrack *fSoALeaks;
  adept::MParray *fLeakedQueue;
  SlotManager *fSlotManager;
};

// Note: deprecated GammaInteractions for split gamma kernels
// struct GammaInteractions {
//   enum Interaction : unsigned int { PairCreation = 0, ComptonScattering = 1, PhotoelectricProcess = 2, NInt };
//   struct Data {
//     double geometryStepLength;
//     double PEmxSec; // Only used for photoelectric process
//     unsigned int slot;
//     vecgeom::NavigationState preStepNavState;
//     vecgeom::Vector3D<Precision> preStepPos;
//     vecgeom::Vector3D<Precision> preStepDir;
//     double preStepEnergy;
//   };
//   adept::MParrayT<Data> *queues[Interaction::NInt];
// };

// A bundle of generators for the three particle types.
struct Secondaries {
  ParticleGenerator electrons;
  ParticleGenerator positrons;
  ParticleGenerator gammas;
};

// Holds the leaked track structs for all three particle types
struct AllLeaked {
  LeakedTracks leakedElectrons;
  LeakedTracks leakedPositrons;
  LeakedTracks leakedGammas;
};

// struct AllSoA {
//   SoATrack *electrons;
//   SoATrack *positrons;
//   SoATrack *gammas;
//   SoATrack *electronsLeaks;
//   SoATrack *positronsLeaks;
//   SoATrack *gammasLeaks;
// };

// A bundle of queues per particle type:
//  * Two for active particles, one for the current iteration and the second for the next.
struct ParticleQueues {
  /*
Gamma interactions:
0 - Conversion
1 - Compton
2 - Photoelectric
3 - Unused
4 - Relocation

Electron interactions:
0 - Ionization
1 - Bremsstrahlung
2 - Unused
3 - Unused
4 - Relocation

Positron interactions:
0 - Ionization
1 - Bremsstrahlung
2 - In flight annihilation
3 - Stopped annihilation
4 - Relocation

In-flight and stopped annihilation use different codes but may be merged to save space
in unused queues or if launching one kernel is faster than two smaller ones

It is not straightforward to allocate just the needed queues per particle type because
ParticleQueues needs to be passed by copy to the kernels, which means that we can't do
dynamic allocations
*/
  static constexpr char numInteractions = 5;
  adept::MParray *nextActive;
  adept::MParray *initiallyActive;
#ifdef USE_SPLIT_KERNELS
  adept::MParray *propagation;
  adept::MParray *interactionQueues[numInteractions];
#endif
  adept::MParray *leakedTracksCurrent;
  adept::MParray *leakedTracksNext;

  void SwapActive() { std::swap(initiallyActive, nextActive); }
  void SwapLeakedQueue() { std::swap(leakedTracksCurrent, leakedTracksNext); }
};

__global__ void ClearOccupiedFlags(unsigned int nItems, short *scanFlags)
{
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < nItems; i += blockDim.x) {
    scanFlags[i] = 0;
  }
}

__global__ void MarkOccupiedSlots(adept::MParray *activeQueue, short *scanFlags)
{
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < activeQueue->size(); i += blockDim.x) {
    scanFlags[(*activeQueue)[i]] = 1;
  }
}

__global__ void CompactCopy(unsigned int nItems, Track *tracks, SoATrack *soaTrack, short *scanFlags, int *scanResult,
                            G4HepEmElectronTrack *electronsHepEm)
{
  // Note: would be more efficient to compute the max occupied index and use it to limit the for loop
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < nItems; i += blockDim.x) {
    // If the slot is occupied, the scan result indicates the destination
    auto dest = scanResult[i];
    if (scanFlags[i] && dest != i) {
      // Copy AdePT Tracks
      tracks[dest] = tracks[i];
      // Copy AdePT SoA Tracks
      soaTrack->fEkin[dest] = soaTrack->fEkin[i];
      // Copy G4HepEm tracks
      electronsHepEm[dest] = electronsHepEm[i];
    }
  }
}

__global__ void UpdateActiveIdx(adept::MParray *activeQueue, SlotManager *slotManager)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("Nused before %ld\n", activeQueue->size());

    activeQueue->fNbooked.store(0);
    activeQueue->fNused.store(slotManager->fSlotCounter);

    printf("Nused updated %ld\n", activeQueue->size());
  }
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < slotManager->fSlotCounter; i += blockDim.x) {
    activeQueue->fData[i] = i;
  }
}

__global__ void ClearSlotManagerStage1(SlotManager *slotManager)
{
  slotManager->PartialClearStage1();
}

__global__ void ClearSlotManagerStage2(SlotManager *slotManager)
{
  slotManager->PartialClearStage2();
}

// Holds all information needed to manage in-flight tracks of one type
struct ParticleType {
  Track *tracks;
  SoATrack *soaTrack;
  Track *leaks;
  SoATrack *soaLeaks;
  SlotManager *slotManager;
  SlotManager *slotManagerLeaks;
  ParticleQueues queues;
  short *scanFlags;
  int *scanResult;
  cudaStream_t stream;
  cudaEvent_t event;
  uint nItems;

  enum {
    Electron = 0,
    Positron = 1,
    Gamma    = 2,

    NumParticleTypes,
  };
  static constexpr double relativeQueueSize[] = {0.35, 0.15, 0.5};

  void Defragment(/*unsigned int nItems, short *scanFlags, int *scanResult*/ G4HepEmElectronTrack *electronsHepEm,
                  cudaStream_t stream)
  {
    // Initialize flags
    // cudaMemset(scanFlags, 0, nItems * sizeof(short));
    ClearOccupiedFlags<<<100, 256, 0, stream>>>(nItems, scanFlags);

    // Mark used slots
    MarkOccupiedSlots<<<100, 256, 0, stream>>>(queues.nextActive, scanFlags);

    // Do the scan
    // thrust::device_ptr<short> flags_thrust(scanFlags);
    // thrust::device_ptr<int> scan_thrust(scanResult);
    // thrust::exclusive_scan(scanFlags, scanFlags + nItems, scanResult);

    thrust::device_ptr<short> flags_thrust(scanFlags);
    thrust::device_ptr<int> scan_thrust(scanResult);
    thrust::exclusive_scan(thrust::cuda::par.on(stream), flags_thrust, flags_thrust + nItems, scan_thrust, 0);

    // Copy each item into to its destination:
    // - Tracks
    // - SoA elements
    // - HepEmTracks
    CompactCopy<<<1, 32, 0, stream>>>(nItems, tracks, soaTrack, scanFlags, scanResult, electronsHepEm);

    // Reset slot manager:
    // - Set fSlotList[fSlotCounter:] to fSlotCounter..nItems
    // - Set fFreeCounter to 0 (We should not need to reinitialize the actual free slots list)
    ClearSlotManagerStage1<<<1000, 32, 0, stream>>>(slotManager);
    ClearSlotManagerStage2<<<1, 1, 0, stream>>>(slotManager);

    // Reset track indices array
    UpdateActiveIdx<<<1000, 32, 0, stream>>>(queues.nextActive, slotManager);
  }
};

#ifdef USE_SPLIT_KERNELS
struct HepEmBuffers {
  G4HepEmElectronTrack *electronsHepEm;
  G4HepEmElectronTrack *positronsHepEm;
  G4HepEmGammaTrack *gammasHepEm;
};

// A bundle of queues per interaction type
struct AllInteractionQueues {
  adept::MParray *queues[5];
};
#endif

// Pointers to track storage for each particle type
struct TracksAndSlots {
  Track *const tracks[ParticleType::NumParticleTypes];
  Track *const leaks[ParticleType::NumParticleTypes];
  SlotManager *const slotManagers[ParticleType::NumParticleTypes];
  SlotManager *const slotManagersLeaks[ParticleType::NumParticleTypes];
};

// A bundle of queues for the three particle types.
struct AllParticleQueues {
  ParticleQueues queues[ParticleType::NumParticleTypes];
};

struct AllSlotManagers {
  SlotManager slotManagers[ParticleType::NumParticleTypes];
  SlotManager slotManagersLeaks[ParticleType::NumParticleTypes];
};

// A data structure to transfer statistics after each iteration.
struct Stats {
  int inFlight[ParticleType::NumParticleTypes];
  int leakedTracks[ParticleType::NumParticleTypes];
  float queueFillLevel[ParticleType::NumParticleTypes];
  float slotFillLevel[ParticleType::NumParticleTypes];
  float slotFillLevelLeaks[ParticleType::NumParticleTypes];
  unsigned int perEventInFlight[kMaxThreads];         // Updated asynchronously
  unsigned int perEventInFlightPrevious[kMaxThreads]; // Used in transport kernels
  unsigned int perEventLeaked[kMaxThreads];
  unsigned int nLeakedCurrent[ParticleType::NumParticleTypes];
  unsigned int nLeakedNext[ParticleType::NumParticleTypes];
  unsigned int hitBufferOccupancy;
};

/// @brief Array of flags whether the event can be finished off
struct AllowFinishOffEventArray {
  unsigned short flags[kMaxThreads];

  __host__ __device__ unsigned short operator[](int idx) const { return flags[idx]; }
};

struct QueueIndexPair {
  unsigned int slot;
  short queue;
};

struct GPUstate {
  ParticleType particles[ParticleType::NumParticleTypes];
  // GammaInteractions gammaInteractions; // Note: deprecated gammaInteractions for split gamma kernels

  std::vector<void *> allCudaPointers;
  // Create a stream to synchronize kernels of all particle types.
  cudaStream_t stream; ///< all-particle sync stream

  static constexpr unsigned int nSlotManager_dev = 3;

  AllSlotManagers allmgr_h;                   // All host slot managers, statically allocated
  SlotManager *slotManager_dev{nullptr};      // All device slot managers
  SlotManager *slotManagerLeaks_dev{nullptr}; // All device leak slot managers

#ifdef USE_SPLIT_KERNELS
  HepEmBuffers hepEmBuffers_d; // All device buffers of hepem tracks
#endif

  Stats *stats_dev{nullptr}; ///< statistics object pointer on device
  Stats *stats{nullptr};     ///< statistics object pointer on host

  PerEventScoring *fScoring_dev; ///< Device array for per-worker scoring data
  std::unique_ptr<HitScoring> fHitScoring;

  adept::MParrayT<QueueIndexPair> *injectionQueue;

  enum class InjectState { Idle, CreatingSlots, ReadyToEnqueue, Enqueueing };
  std::atomic<InjectState> injectState;
  // ExtractState:
  // Idle: No flush has been requested
  // ExtractionRequested: An event requested a flush, waiting for transport to finish
  // TracksNeedTransfer: An event requested a flush, leak buffer on device has tracks to transfer
  // PreparingTracks: Tracks are being copied to the staging buffer
  // TracksReadyToCopy: Staging buffer is ready to be copied to host
  // CopyingTracks: Tracks are being copied to host
  // TracksOnHost: Some or all the tracks have been transferred from device to host and are waiting in the copy buffer
  // SavingTracks: Tracks are being copied to per-event queues
  // TracksSaved: Tracks have been moved from the copy buffer to their respective per-event queues
  enum class ExtractState {
    Idle,
    ExtractionRequested,
    TracksNeedTransfer,
    PreparingTracks,
    TracksReadyToCopy,
    CopyingTracks,
    TracksOnHost,
    SavingTracks,
    TracksSaved
  };
  std::atomic<ExtractState> extractState;
  std::atomic_bool runTransport{true}; ///< Keep transport thread running

  ~GPUstate()
  {
    if (stats) COPCORE_CUDA_CHECK(cudaFreeHost(stats));
    if (stream) COPCORE_CUDA_CHECK(cudaStreamDestroy(stream));

    for (ParticleType &particleType : particles) {
      if (particleType.stream) COPCORE_CUDA_CHECK(cudaStreamDestroy(particleType.stream));
      if (particleType.event) COPCORE_CUDA_CHECK(cudaEventDestroy(particleType.event));
    }
    for (void *ptr : allCudaPointers) {
      COPCORE_CUDA_CHECK(cudaFree(ptr));
    }
    allCudaPointers.clear();
  }
};

// Implementation of the GPUstate deleter
void GPUstateDeleter::operator()(GPUstate *ptr)
{
  delete ptr;
}

// Constant data structures from G4HepEm accessed by the kernels.
// (defined in TestEm3.cu)
extern __constant__ __device__ struct G4HepEmParameters g4HepEmPars;
extern __constant__ __device__ struct G4HepEmData g4HepEmData;

// Pointer for array of volume auxiliary data on device
extern __constant__ __device__ adeptint::VolAuxData *gVolAuxData;

constexpr double kPush = 1.e-8 * copcore::units::cm;
#ifdef ADEPT_USE_EXT_BFIELD
__device__ GeneralMagneticField *gMagneticField = nullptr;
#else
__device__ UniformMagneticField *gMagneticField = nullptr;
#endif

} // namespace AsyncAdePT

#endif
