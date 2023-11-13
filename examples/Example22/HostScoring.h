// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef HOSTSCORING_H
#define HOSTSCORING_H

#include "VecGeom/navigation/NavigationState.h"
#include <AdePT/Atomic.h>
#include <G4ios.hh>

struct GPUStepPoint {
  vecgeom::Vector3D<Precision> fPosition;
  vecgeom::Vector3D<Precision> fMomentumDirection;
  vecgeom::Vector3D<Precision> fPolarization;
  double fEKin;
  double fCharge;
  // Data needed to reconstruct G4 Touchable history
  NavIndex_t fNavigationStateIndex{0}; // VecGeom navigation state index, used to identify the touchable
};

// Stores the necessary data to reconstruct GPU hits on the host , and
// call the user-defined Geant4 sensitive detector code
struct GPUHit {
  char fParticleType{0};                  // Particle type ID
  // Data needed to reconstruct G4 Step
  double fStepLength{0};
  double fTotalEnergyDeposit{0};
  double fNonIonizingEnergyDeposit{0};
  // bool fFirstStepInVolume{false};
  // bool fLastStepInVolume{false};
  // Data needed to reconstruct pre-post step points
  GPUStepPoint fPreStepPoint;
  GPUStepPoint fPostStepPoint;
};

// Provides methods for recording GPU hits, and moving them between host and device
struct HostScoring {

  /// @brief The data in this struct is copied from device to host after each iteration
  struct Stats {
    unsigned int fUsedSlots;   ///< Number of used hit slots
    unsigned int fNextFreeHit; ///< Index of last used hit slot in the buffer
    unsigned int fBufferStart; ///< Index of first used hit slot in the buffer
  };

  HostScoring(unsigned int aBufferCapacity = 1024 * 1024, float aFlushLimit = 0.8)
      : fBufferCapacity(aBufferCapacity), fFlushLimit(aFlushLimit)
  {
    printf("Initializing scoring with buffer capacity: %d\n", aBufferCapacity);
    // Allocate the hits buffer on Host
    fGPUHitsBuffer_host = (GPUHit *)malloc(sizeof(GPUHit) * fBufferCapacity);
  };

  ~HostScoring() = default;

  /// @brief Allocate and initialize data structures on device
  HostScoring *InitializeOnGPU();

  void FreeGPU(HostScoring *aHostScoring_dev);

  /// @brief Record a hit
  __device__ void RecordHit(char aParticleType, double aStepLength,
                            double aTotalEnergyDeposit, 
                            vecgeom::NavigationState const *aPreState,
                            vecgeom::Vector3D<Precision> *aPrePosition,
                            vecgeom::Vector3D<Precision> *aPreMomentumDirection,
                            vecgeom::Vector3D<Precision> *aPrePolarization, double aPreEKin, double aPreCharge,
                            vecgeom::NavigationState const *aPostState,
                            vecgeom::Vector3D<Precision> *aPostPosition,
                            vecgeom::Vector3D<Precision> *aPostMomentumDirection,
                            vecgeom::Vector3D<Precision> *aPostPolarization, double aPostEKin, double aPostCharge);

  /// @brief Get index of the next free hit slot
  __device__ __forceinline__ unsigned int GetNextFreeHitIndex();

  /// @brief Get reference to the next free hit struct in the buffer
  __device__ __forceinline__ GPUHit *GetNextFreeHit();

  /// @brief Utility function to copy a 3D vector, used for filling the Step Points
  __device__ __forceinline__ void Copy3DVector(vecgeom::Vector3D<Precision> *source,
                                               vecgeom::Vector3D<Precision> *destination);

  /// @brief Check if the buffer is filled over a certain capacity and copy the hits to the host if so
  /// @return True if the buffer was transferred to the host
  bool CheckAndFlush(Stats &aStats_host, HostScoring *aScoring_device, cudaStream_t stream);

  /// @brief Copy the hits buffer to the host
  void CopyHitsToHost(Stats &aStats_host, HostScoring *aScoring_device, cudaStream_t stream);

  /// @brief Update the stats struct. To be called before copying it back.
  /// This is an in-device update, meant to copy the state of the member variables we need into a struct
  /// that can be copied back to the host
  __device__ __forceinline__ void refresh_stats()
  {
    fStats.fUsedSlots   = fUsedSlots_d->load();
    fStats.fNextFreeHit = fNextFreeHit_d->load();
    fStats.fBufferStart = fBufferStart;

    /*
    printf("DEBUG: Buffer Start: %u\n", fBufferStart);
    printf("DEBUG: Buffer Usage: %u\n", fUsedSlots_d->load());
    printf("DEBUG: Next Free Hit: %u\n", fNextFreeHit_d->load());
    printf("DEBUG: Buffer Capacity: %u\n", fBufferCapacity);
    */
  }

  /// @brief Print scoring info
  void Print();

  // Data members
  unsigned int fBufferCapacity{0}; ///< Number of hits to be stored in the buffer
  float fFlushLimit{0};            ///< Proportion of the buffer that needs to be filled to trigger a flush to CPU
  unsigned int fBufferStart{0};    ///< Index of first used slot in the buffer
  GPUHit *fGPUHitsBuffer_device{nullptr};
  GPUHit *fGPUHitsBuffer_host{nullptr};

  // Atomic variables used on GPU
  adept::Atomic_t<unsigned int> *fUsedSlots_d;   ///< Number of used hit slots
  adept::Atomic_t<unsigned int> *fNextFreeHit_d; ///< Index of last used hit slot in the buffer

  // Stats struct, used to transfer information about the state of the buffer
  Stats fStats;

  // Used to get performance information
  unsigned int fStepsSinceLastFlush{0};
};

#endif // HOSTSCORING_H