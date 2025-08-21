// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>

using VolAuxData = adeptint::VolAuxData;

namespace AsyncAdePT {

// __device__ __forceinline__ void CopyTrack(int slotSrc, int slotDst, Track *src, Track *dst, SoATrack *soaSrc,
//                                           SoATrack *soaDst)
// {
//   dst[slotDst]                = src[slotSrc];
//   soaDst->fEkin[slotDst]      = soaSrc->fEkin[slotSrc];
//   soaDst->fSafety[slotDst]    = soaSrc->fSafety[slotSrc];
//   soaDst->fSafetyPos[slotDst] = soaSrc->fSafetyPos[slotSrc];
// }

__global__ void GammaHowFar(Track *gammas, SoATrack *soaTrack, Track *leaks, SoATrack *soaLeaks,
                            G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active, Secondaries secondaries,
                            adept::MParray *propagationQueue, adept::MParray *leakedQueue, Stats *InFlightStats,
                            AllowFinishOffEventArray allowFinishOffEvent)
{
  constexpr unsigned short maxSteps        = 10'000;
  constexpr unsigned short kStepsStuckKill = 25;
  int activeSize                           = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];
    // auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

    // gammas[slot].currentSlot = slot;

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    currentTrack.preStepEKin = soaTrack->fEkin[slot];
    currentTrack.preStepPos  = currentTrack.pos;
    currentTrack.preStepDir  = currentTrack.dir;
    // the MCC vector is indexed by the logical volume id

    currentTrack.stepCounter++;
    bool printErrors = false;
    if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
      if (printErrors)
        printf("Killing gamma event %d track %lu E=%f lvol=%d after %d steps with zeroStepCounter %u. This indicates a "
               "stuck particle!\n",
               currentTrack.eventId, currentTrack.trackId, soaTrack->fEkin[slot], lvolID, currentTrack.stepCounter,
               currentTrack.zeroStepCounter);
      continue;
    }

    // Write local variables back into track and enqueue
    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot = secondaries.gammas.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = gammas[slot];
        CopyTrack(slot, leakSlot, gammas, leaks, soaTrack, soaLeaks);
        auto success = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        // slotManager.MarkSlotForFreeing(slot);
      }
    };

    if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
        InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
      printf("Thread %d Finishing gamma of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
             currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
             currentTrack.eventId, soaTrack->fEkin[slot], lvolID, currentTrack.stepCounter);
      survive(LeakStatus::FinishEventOnCPU);
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    gammaTrack.ReSet();
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(soaTrack->fEkin[slot]);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Re-sample the `number-of-interaction-left` (if needed, otherwise use stored numIALeft) and put it into the
    // G4HepEmTrack. Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section. The
    // currentTrack.numIALeft[0] are updated later
    if (currentTrack.numIALeft[0] <= 0.0) {
      theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), 0);
    } else {
      theTrack->SetNumIALeft(currentTrack.numIALeft[0], 0);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Particles that were not cut or leaked are added to the queue used by the next kernels
    propagationQueue->push_back(slot);
  }
}

__global__ void GammaPropagation(Track *gammas, SoATrack *soaTrack, G4HepEmGammaTrack *hepEMTracks,
                                 const adept::MParray *active)
{
  constexpr Precision kPushDistance        = 1000 * vecgeom::kTolerance;
  constexpr Precision kPushStuck           = 100 * vecgeom::kTolerance;
  constexpr unsigned short kStepsStuckPush = 5;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = gammas[slot];

    int lvolID = currentTrack.navState.GetLogicalId();

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Check if there's a volume boundary in between.
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    currentTrack.geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, currentTrack.nextState, hitsurf_index);
#else
    currentTrack.geometryStepLength =
        AdePTNavigator::ComputeStepAndNextVolume(currentTrack.pos, currentTrack.dir, theTrack->GetGStepLength(),
                                                 currentTrack.navState, currentTrack.nextState, kPushDistance);
#endif
    //  printf("pvol=%d  step=%g  onboundary=%d  pos={%g, %g, %g}  dir={%g, %g, %g}\n", navState.TopId(),
    //  geometryStepLength,
    //         nextState.IsOnBoundary(), pos[0], pos[1], pos[2], dir[0], dir[1], dir[2]);

    if (currentTrack.geometryStepLength < kPushStuck && currentTrack.geometryStepLength < theTrack->GetGStepLength()) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) currentTrack.pos += kPushStuck * currentTrack.dir;
    } else
      currentTrack.zeroStepCounter = 0;

    currentTrack.pos += currentTrack.geometryStepLength * currentTrack.dir;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    currentTrack.navState.SetBoundaryState(currentTrack.nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(currentTrack.geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    currentTrack.globalTime += deltaTime;
    currentTrack.localTime += deltaTime;
  }
}

template <typename Scoring>
__global__ void GammaSetupInteractions(Track *gammas, SoATrack *soaTrack, Track *leaks, SoATrack *soaLeaks,
                                       G4HepEmGammaTrack *hepEMTracks, const adept::MParray *active,
                                       Secondaries secondaries, adept::MParray *nextActiveQueue,
                                       AllInteractionQueues interactionQueues, adept::MParray *leakedQueue,
                                       Scoring *userScoring)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];
    // auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

    int lvolID = currentTrack.navState.GetLogicalId();

    // Write local variables back into track and enqueue
    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot = secondaries.gammas.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = gammas[slot];
        CopyTrack(slot, leakSlot, gammas, leaks, soaTrack, soaLeaks);
        auto success = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        // slotManager.MarkSlotForFreeing(slot);
      } else {
        // Get a slot in the next active queue
        // This is necessary as we are copying the track to the next array
        auto nextSlot = secondaries.gammas.NextSlot();
        // Copy the track to the next active tracks array
        CopyTrack(slot, nextSlot, gammas, secondaries.gammas.fNextTracks, soaTrack, secondaries.gammas.fSoANextTracks);
        // Store the slot in the next active queue
        nextActiveQueue->push_back(nextSlot);
      }
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    if (currentTrack.nextState.IsOnBoundary()) {
      interactionQueues.queues[4]->push_back(slot);
      continue;
    } else {
      currentTrack.restrictedPhysicalStepLength = false;

      // NOTE: This may be moved to the next kernel
      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());

      // Reset number of interaction left for the winner discrete process also in the currentTrack
      // (SampleInteraction() resets it for theTrack), will be resampled in the next iteration.
      currentTrack.numIALeft[0] = -1.0;

      if (theTrack->GetWinnerProcessIndex() < 3) {
        interactionQueues.queues[theTrack->GetWinnerProcessIndex()]->push_back(slot);
      } else {
        // IMPORTANT: This is necessary just for getting numerically identical results,
        // but should be removed once confirmed that results are good
        G4HepEmRandomEngine rnge(&currentTrack.rngState);
        RanluxppDouble newRNG(currentTrack.rngState.Branch());
        // Gamma nuclear needs to be handled by Geant4 directly, passing track back to CPU
        survive(LeakStatus::GammaNuclear);
      }
    }
  }
}

template <typename Scoring>
__global__ void GammaRelocation(Track *gammas, SoATrack *soaTrack, Track *leaks, SoATrack *soaLeaks,
                                G4HepEmGammaTrack *hepEMTracks, Secondaries secondaries,
                                adept::MParray *nextActiveQueue, adept::MParray *relocatingQueue,
                                adept::MParray *leakedQueue, Scoring *userScoring, const bool returnAllSteps,
                                const bool returnLastStep)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  int activeSize                    = relocatingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*relocatingQueue)[i];
    // auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

    int lvolID = currentTrack.navState.GetLogicalId();

    // Write local variables back into track and enqueue
    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      currentTrack.leakStatus = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot = secondaries.gammas.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = gammas[slot];
        CopyTrack(slot, leakSlot, gammas, leaks, soaTrack, soaLeaks);
        auto success = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        // slotManager.MarkSlotForFreeing(slot);
      } else {
        // Get a slot in the next active queue
        // This is necessary as we are copying the track to the next array
        auto nextSlot = secondaries.gammas.NextSlot();
        // Copy the track to the next active tracks array
        CopyTrack(slot, nextSlot, gammas, secondaries.gammas.fNextTracks, soaTrack, secondaries.gammas.fSoANextTracks);
        // Store the slot in the next active queue
        nextActiveQueue->push_back(nextSlot);
      }
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    currentTrack.restrictedPhysicalStepLength = true;
    // For now, just count that we hit something.

    // Kill the particle if it left the world.
    if (!currentTrack.nextState.IsOutside()) {

      G4HepEmGammaManager::UpdateNumIALeft(theTrack);

      // Save the `number-of-interaction-left` in our track.
      // Use index 0 since numIALeft stores for gammas only the total macroscopic cross section
      double numIALeft          = theTrack->GetNumIALeft(0);
      currentTrack.numIALeft[0] = numIALeft;

#ifdef ADEPT_USE_SURF
      AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, hitsurf_index, currentTrack.nextState);
#else
      AdePTNavigator::RelocateToNextVolume(currentTrack.pos, currentTrack.dir, currentTrack.nextState);
#endif

      // if all steps are returned, we need to record the hit here,
      // as now the nextState is defined, but the navState is not yet replaced
      if (returnAllSteps)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.trackId,                        // Track ID
                                 currentTrack.parentId,                       // parent Track ID
                                 static_cast<short>(10),                      // step defining process ID
                                 2,                                           // Particle type
                                 currentTrack.geometryStepLength,             // Step length
                                 0,                                           // Total Edep
                                 currentTrack.weight,                         // Track weight
                                 currentTrack.navState,                       // Pre-step point navstate
                                 currentTrack.preStepPos,                     // Pre-step point position
                                 currentTrack.preStepDir,                     // Pre-step point momentum direction
                                 currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                                 currentTrack.nextState,                      // Post-step point navstate
                                 currentTrack.pos,                            // Post-step point position
                                 currentTrack.dir,                            // Post-step point momentum direction
                                 soaTrack->fEkin[slot],                       // Post-step point kinetic energy
                                 currentTrack.globalTime,                     // global time
                                 currentTrack.localTime,                      // local time
                                 currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                 false,                     // whether this is the last step of the track
                                 currentTrack.stepCounter); // stepcounter

      // Move to the next boundary.
      currentTrack.navState = currentTrack.nextState;
      // printf("  -> pvol=%d pos={%g, %g, %g} \n", navState.TopId(), pos[0], pos[1], pos[2]);
      //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
      const int nextlvolID          = currentTrack.navState.GetLogicalId();
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
      if (nextauxData.fGPUregion > 0) {
        survive();
      } else {
        // To be safe, just push a bit the track exiting the GPU region to make sure
        // Geant4 does not relocate it again inside the same region
        currentTrack.pos += kPushDistance * currentTrack.dir;
        survive(LeakStatus::OutOfGPURegion);
      }
    } else {
      // release slot for particle that has left the world
      // slotManager.MarkSlotForFreeing(slot);

      // particle has left the world, record hit if last or all steps are returned
      if (returnAllSteps || returnLastStep)
        adept_scoring::RecordHit(userScoring,
                                 currentTrack.trackId,                        // Track ID
                                 currentTrack.parentId,                       // parent Track ID
                                 static_cast<short>(10),                      // step defining process ID
                                 2,                                           // Particle type
                                 currentTrack.geometryStepLength,             // Step length
                                 0,                                           // Total Edep
                                 currentTrack.weight,                         // Track weight
                                 currentTrack.navState,                       // Pre-step point navstate
                                 currentTrack.preStepPos,                     // Pre-step point position
                                 currentTrack.preStepDir,                     // Pre-step point momentum direction
                                 currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                                 currentTrack.nextState,                      // Post-step point navstate
                                 currentTrack.pos,                            // Post-step point position
                                 currentTrack.dir,                            // Post-step point momentum direction
                                 soaTrack->fEkin[slot],                       // Post-step point kinetic energy
                                 currentTrack.globalTime,                     // global time
                                 currentTrack.localTime,                      // local time
                                 currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                 true,                      // whether this is the last step of the track
                                 currentTrack.stepCounter); // stepcounter
    }
    continue;
  }
}

template <typename Scoring>
__global__ void GammaConversion(Track *gammas, SoATrack *soaTrack, G4HepEmGammaTrack *hepEMTracks,
                                Secondaries secondaries, adept::MParray *nextActiveQueue,
                                adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                const bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot = (*interactingQueue)[i];
    // auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    // Write local variables back into track and enqueue
    auto survive = [&]() {
      isLastStep = false; // particle survived
      // Get a slot in the next active queue
      // This is necessary as we are copying the track to the next array
      auto nextSlot = secondaries.gammas.NextSlot();
      // Copy the track to the next active tracks array
      CopyTrack(slot, nextSlot, gammas, secondaries.gammas.fNextTracks, soaTrack, secondaries.gammas.fSoANextTracks);
      // Store the slot in the next active queue
      nextActiveQueue->push_back(nextSlot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double thePosCut   = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecPosProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    // Interaction

    // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
    if (soaTrack->fEkin[slot] < 2 * copcore::units::kElectronMassC2) {
      survive();
      break;
    }

    double logEnergy = std::log(soaTrack->fEkin[slot]);
    double elKinEnergy, posKinEnergy;
    G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, soaTrack->fEkin[slot], logEnergy,
                                                         auxData.fMCIndex, elKinEnergy, posKinEnergy, &rnge);

    double dirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirSecondaryEl[3], dirSecondaryPos[3];
    G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                        posKinEnergy, &rnge);

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

    // Check the cuts and deposit energy in this volume if needed
    if (ApplyCuts && elKinEnergy < theElCut) {
      // Deposit the energy here and kill the secondary
      edep = elKinEnergy;
    } else {
      Track &electron = secondaries.electrons.NextTrack(
          elKinEnergy, newRNG, currentTrack.pos,
          vecgeom::Vector3D<Precision>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                 /*CreatorProcessId*/ short(0),
                                 /* electron*/ 0,                     // Particle type
                                 0,                                   // Step length
                                 0,                                   // Total Edep
                                 electron.weight,                     // Track weight
                                 electron.navState,                   // Pre-step point navstate
                                 electron.pos,                        // Pre-step point position
                                 electron.dir,                        // Pre-step point momentum direction
                                 elKinEnergy,                         // Pre-step point kinetic energy
                                 electron.navState,                   // Post-step point navstate
                                 electron.pos,                        // Post-step point position
                                 electron.dir,                        // Post-step point momentum direction
                                 elKinEnergy,                         // Post-step point kinetic energy
                                 electron.globalTime,                 // global time
                                 0.,                                  // local time
                                 electron.eventId, electron.threadId, // eventID and threadID
                                 false,                               // whether this was the last step
                                 electron.stepCounter);               // whether this was the first step
      }
    }

    if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < thePosCut)) {
      // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
      edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
    } else {
      Track &positron = secondaries.positrons.NextTrack(
          posKinEnergy, currentTrack.rngState, currentTrack.pos,
          vecgeom::Vector3D<Precision>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]},
          currentTrack.navState, currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, positron.trackId, positron.parentId,
                                 /*CreatorProcessId*/ short(0),
                                 /* positron*/ 1,                     // Particle type
                                 0,                                   // Step length
                                 0,                                   // Total Edep
                                 positron.weight,                     // Track weight
                                 positron.navState,                   // Pre-step point navstate
                                 positron.pos,                        // Pre-step point position
                                 positron.dir,                        // Pre-step point momentum direction
                                 posKinEnergy,                        // Pre-step point kinetic energy
                                 positron.navState,                   // Post-step point navstate
                                 positron.pos,                        // Post-step point position
                                 positron.dir,                        // Post-step point momentum direction
                                 posKinEnergy,                        // Post-step point kinetic energy
                                 positron.globalTime,                 // global time
                                 0.,                                  // local time
                                 positron.eventId, positron.threadId, // eventID and threadID
                                 false,                               // whether this was the last step
                                 positron.stepCounter);               // whether this was the first step
      }
    }

    // The current track is killed by not enqueuing into the next activeQueue and the slot is released
    // slotManager.MarkSlotForFreeing(slot);

    //////////////

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(0),                       // step defining process ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               currentTrack.globalTime,                     // global time
                               currentTrack.localTime,                      // local time
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               isLastStep,                // whether this is the last step of the track
                               currentTrack.stepCounter); // stepcounter
    }
  }
}

template <typename Scoring>
__global__ void GammaCompton(Track *gammas, SoATrack *soaTrack, G4HepEmGammaTrack *hepEMTracks, Secondaries secondaries,
                             adept::MParray *nextActiveQueue, adept::MParray *interactingQueue, Scoring *userScoring,
                             const bool returnAllSteps, const bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot = (*interactingQueue)[i];
    // auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    // Write local variables back into track and enqueue
    auto survive = [&]() {
      isLastStep = false; // particle survived
      // Get a slot in the next active queue
      // This is necessary as we are copying the track to the next array
      auto nextSlot = secondaries.gammas.NextSlot();
      // Copy the track to the next active tracks array
      CopyTrack(slot, nextSlot, gammas, secondaries.gammas.fNextTracks, soaTrack, secondaries.gammas.fSoANextTracks);
      // Store the slot in the next active queue
      nextActiveQueue->push_back(nextSlot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    // Interaction

    // Invoke Compton scattering of gamma.
    constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
    if (soaTrack->fEkin[slot] < LowEnergyThreshold) {
      survive();
      break;
    }
    const double origDirPrimary[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
    double dirPrimary[3];
    newEnergyGamma = G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(soaTrack->fEkin[slot], dirPrimary,
                                                                                    origDirPrimary, &rnge);
    vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

    const double energyEl = soaTrack->fEkin[slot] - newEnergyGamma;

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

    // Check the cuts and deposit energy in this volume if needed
    if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
      // Create a secondary electron and sample/compute directions.
      Track &electron = secondaries.electrons.NextTrack(
          energyEl, newRNG, currentTrack.pos, soaTrack->fEkin[slot] * currentTrack.dir - newEnergyGamma * newDirGamma,
          currentTrack.navState, currentTrack, currentTrack.globalTime);
      electron.dir.Normalize();

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                 /*CreatorProcessId*/ short(1),
                                 /* electron*/ 0,                     // Particle type
                                 0,                                   // Step length
                                 0,                                   // Total Edep
                                 electron.weight,                     // Track weight
                                 electron.navState,                   // Pre-step point navstate
                                 electron.pos,                        // Pre-step point position
                                 electron.dir,                        // Pre-step point momentum direction
                                 energyEl,                            // Pre-step point kinetic energy
                                 electron.navState,                   // Post-step point navstate
                                 electron.pos,                        // Post-step point position
                                 electron.dir,                        // Post-step point momentum direction
                                 energyEl,                            // Post-step point kinetic energy
                                 electron.globalTime,                 // global time
                                 0.,                                  // local time
                                 electron.eventId, electron.threadId, // eventID and threadID
                                 false,                               // whether this was the last step
                                 electron.stepCounter);               // whether this was the first step
      }

    } else {
      edep = energyEl;
    }

    // Check the new gamma energy and deposit if below threshold.
    // Using same hardcoded very LowEnergyThreshold as G4HepEm
    if (newEnergyGamma > LowEnergyThreshold) {
      soaTrack->fEkin[slot] = newEnergyGamma;
      currentTrack.dir      = newDirGamma;
      survive();
    } else {
      edep += newEnergyGamma;
      newEnergyGamma = 0.;
      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
      // slotManager.MarkSlotForFreeing(slot);
    }

    //////////////

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(1),                       // step defining process ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               currentTrack.globalTime,                     // global time
                               currentTrack.localTime,                      // local time
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               isLastStep, // whether this is the last step of the track
                               currentTrack.stepCounter);
    }
  }
}

template <typename Scoring>
__global__ void GammaPhotoelectric(Track *gammas, SoATrack *soaTrack, G4HepEmGammaTrack *hepEMTracks,
                                   Secondaries secondaries, adept::MParray *nextActiveQueue,
                                   adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                   const bool returnLastStep)
{
  // int activeSize = active->size();
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot      = (*active)[i];
    const int slot = (*interactingQueue)[i];
    // auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];

    int lvolID                = currentTrack.navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    // Write local variables back into track and enqueue
    auto survive = [&]() {
      isLastStep = false; // particle survived
      // Get a slot in the next active queue
      // This is necessary as we are copying the track to the next array
      auto nextSlot = secondaries.gammas.NextSlot();
      // Copy the track to the next active tracks array
      CopyTrack(slot, nextSlot, gammas, secondaries.gammas.fNextTracks, soaTrack, secondaries.gammas.fSoANextTracks);
      // Store the slot in the next active queue
      nextActiveQueue->push_back(nextSlot);
    };

    G4HepEmGammaTrack &gammaTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = gammaTrack.GetTrack();

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    // Interaction

    // Invoke photoelectric process.
    const double theLowEnergyThreshold = 1 * copcore::units::eV;

    const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
        &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), soaTrack->fEkin[slot], &rnge);

    edep                    = bindingEnergy;
    const double photoElecE = soaTrack->fEkin[slot] - edep;
    if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      double dirGamma[] = {currentTrack.dir.x(), currentTrack.dir.y(), currentTrack.dir.z()};
      double dirPhotoElec[3];
      G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

      // Create a secondary electron and sample directions.
      Track &electron = secondaries.electrons.NextTrack(
          photoElecE, newRNG, currentTrack.pos,
          vecgeom::Vector3D<Precision>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]}, currentTrack.navState,
          currentTrack, currentTrack.globalTime);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                 /*CreatorProcessId*/ short(2),
                                 /* electron*/ 0,                     // Particle type
                                 0,                                   // Step length
                                 0,                                   // Total Edep
                                 electron.weight,                     // Track weight
                                 electron.navState,                   // Pre-step point navstate
                                 electron.pos,                        // Pre-step point position
                                 electron.dir,                        // Pre-step point momentum direction
                                 photoElecE,                          // Pre-step point kinetic energy
                                 electron.navState,                   // Post-step point navstate
                                 electron.pos,                        // Post-step point position
                                 electron.dir,                        // Post-step point momentum direction
                                 photoElecE,                          // Post-step point kinetic energy
                                 electron.globalTime,                 // global time
                                 0.,                                  // local time
                                 electron.eventId, electron.threadId, // eventID and threadID
                                 false,                               // whether this was the last step
                                 electron.stepCounter);               // whether this was the first step
      }

    } else {
      // If the secondary electron is cut, deposit all the energy of the gamma in this volume
      edep = soaTrack->fEkin[slot];
    }
    // The current track is killed by not enqueuing into the next activeQueue and the slot is released
    // slotManager.MarkSlotForFreeing(slot);

    //////////////

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               static_cast<short>(2),                       // step defining process ID
                               2,                                           // Particle type
                               currentTrack.geometryStepLength,             // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               currentTrack.navState,                       // Pre-step point navstate
                               currentTrack.preStepPos,                     // Pre-step point position
                               currentTrack.preStepDir,                     // Pre-step point momentum direction
                               currentTrack.preStepEKin,                    // Pre-step point kinetic energy
                               currentTrack.nextState,                      // Post-step point navstate
                               currentTrack.pos,                            // Post-step point position
                               currentTrack.dir,                            // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               currentTrack.globalTime,                     // global time
                               currentTrack.localTime,                      // local time
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               isLastStep, // whether this is the last step of the track
                               currentTrack.stepCounter);
    }
  }
}

} // namespace AsyncAdePT
