// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

// Classes for Runge-Kutta integration
#include <AdePT/magneticfield/MagneticFieldEquation.h>
#include <AdePT/magneticfield/DormandPrinceRK45.h>
#include <AdePT/magneticfield/fieldPropagatorRungeKutta.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/AdePTPrecision.hh>

#include <G4HepEmElectronManager.hh>
#include <G4HepEmElectronTrack.hh>
#include <G4HepEmElectronInteractionBrem.hh>
#include <G4HepEmElectronInteractionIoni.hh>
#include <G4HepEmElectronInteractionUMSC.hh>
#include <G4HepEmPositronInteractionAnnihilation.hh>
// Pull in implementation.
#include <G4HepEmRunUtils.icc>
#include <G4HepEmInteractionUtils.icc>
#include <G4HepEmElectronManager.icc>
#include <G4HepEmElectronInteractionBrem.icc>
#include <G4HepEmElectronInteractionIoni.icc>
#include <G4HepEmElectronInteractionUMSC.icc>
#include <G4HepEmPositronInteractionAnnihilation.icc>
#include <G4HepEmElectronEnergyLossFluctuation.icc>

using VolAuxData = adeptint::VolAuxData;

// Compute velocity based on the kinetic energy of the particle
__device__ double GetVelocity(double eKin)
{
  // Taken from G4DynamicParticle::ComputeBeta
  double T    = eKin / copcore::units::kElectronMassC2;
  double beta = sqrt(T * (T + 2.)) / (T + 1.0);
  return copcore::units::kCLight * beta;
}

namespace AsyncAdePT {

__device__ __forceinline__ void CopyTrack(int slotSrc, int slotDst, Track *src, Track *dst, SoATrack *soaSrc,
                                          SoATrack *soaDst)
{
  dst[slotDst]                         = src[slotSrc];
  soaDst->fEkin[slotDst]               = soaSrc->fEkin[slotSrc];
  soaDst->fSafety[slotDst]             = soaSrc->fSafety[slotSrc];
  soaDst->fSafetyPos[slotDst]          = soaSrc->fSafetyPos[slotSrc];
  soaDst->fPos[slotDst]                = soaSrc->fPos[slotSrc];
  soaDst->fDir[slotDst]                = soaSrc->fDir[slotSrc];
  soaDst->fGlobalTime[slotDst]         = soaSrc->fGlobalTime[slotSrc];
  soaDst->fLocalTime[slotDst]          = soaSrc->fLocalTime[slotSrc];
  soaDst->fProperTime[slotDst]         = soaSrc->fProperTime[slotSrc];
  soaDst->fNavState[slotDst]           = soaSrc->fNavState[slotSrc];
  soaDst->fOriginNavState[slotDst]     = soaSrc->fOriginNavState[slotSrc];
  soaDst->fWeight[slotDst]             = soaSrc->fWeight[slotSrc];
  soaDst->fThreadId[slotDst]           = soaSrc->fThreadId[slotSrc];
  soaDst->fParentId[slotDst]           = soaSrc->fParentId[slotSrc];
  soaDst->fEventId[slotDst]            = soaSrc->fEventId[slotSrc];
  soaDst->fRngState[slotDst]           = soaSrc->fRngState[slotSrc];
  soaDst->fTrackId[slotDst]            = soaSrc->fTrackId[slotSrc];
  soaDst->fStepCounter[slotDst]        = soaSrc->fStepCounter[slotSrc];
  soaDst->fLooperCounter[slotDst]      = soaSrc->fLooperCounter[slotSrc];
  soaDst->fZeroStepCounter[slotDst]    = soaSrc->fZeroStepCounter[slotSrc];
  soaDst->fNumIALeft[slotDst]          = soaSrc->fNumIALeft[slotSrc];
  soaDst->fInitialRange[slotDst]       = soaSrc->fInitialRange[slotSrc];
  soaDst->fDynamicRangeFactor[slotDst] = soaSrc->fDynamicRangeFactor[slotSrc];
  soaDst->fTlimitMin[slotDst]          = soaSrc->fTlimitMin[slotSrc];
  soaDst->fLeakStatus[slotDst]         = soaSrc->fLeakStatus[slotSrc];
}

template <bool IsElectron>
__global__ void ElectronHowFar(Track *electrons, SoATrack *soaTrack, Track *leaks, SoATrack *soaLeaks,
                               G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active, Secondaries secondaries,
                               adept::MParray *nextActiveQueue, adept::MParray *propagationQueue,
                               adept::MParray *leakedQueue, Stats *InFlightStats,
                               AllowFinishOffEventArray allowFinishOffEvent)
{
  constexpr unsigned short maxSteps        = 10'000;
  constexpr int Charge                     = IsElectron ? -1 : 1;
  constexpr double restMass                = copcore::units::kElectronMassC2;
  constexpr int Nvar                       = 6;
  constexpr unsigned short kStepsStuckKill = 25;

#ifdef ADEPT_USE_EXT_BFIELD
  using Field_t = GeneralMagneticField;
#else
  using Field_t = UniformMagneticField;
#endif
  using Equation_t = MagneticFieldEquation<Field_t>;
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, rk_integration_t>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, rk_integration_t, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    Track &currentTrack = electrons[slot];
    // SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager :
    // *secondaries.positrons.fSlotManager;

    // electrons[slot].currentSlot = slot;

    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    currentTrack.preStepEKin = soaTrack->fEkin[slot];
    currentTrack.preStepPos  = soaTrack->fPos[slot];
    currentTrack.preStepDir  = soaTrack->fDir[slot];
    soaTrack->fStepCounter[slot]++;
    bool printErrors = false;
    if (soaTrack->fStepCounter[slot] >= maxSteps || soaTrack->fZeroStepCounter[slot] > kStepsStuckKill) {
      if (printErrors)
        printf("Killing e-/+ event %d track %ld E=%f lvol=%d after %d steps with zeroStepCounter %u\n",
               soaTrack->fEventId[slot], soaTrack->fTrackId[slot], soaTrack->fEkin[slot], lvolID,
               soaTrack->fStepCounter[slot], soaTrack->fZeroStepCounter[slot]);
      continue;
    }

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      soaTrack->fLeakStatus[slot] = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot;
        if (IsElectron)
          leakSlot = secondaries.electrons.NextLeakSlot();
        else
          leakSlot = secondaries.positrons.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = electrons[slot];
        CopyTrack(slot, leakSlot, electrons, leaks, soaTrack, soaLeaks);
        auto success = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        // slotManager.MarkSlotForFreeing(slot);
      }
    };

    if (InFlightStats->perEventInFlightPrevious[soaTrack->fThreadId[slot]] <
            allowFinishOffEvent[soaTrack->fThreadId[slot]] &&
        InFlightStats->perEventInFlightPrevious[soaTrack->fThreadId[slot]] != 0) {
      printf("Thread %d Finishing e-/e+ of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
             soaTrack->fThreadId[slot], InFlightStats->perEventInFlightPrevious[soaTrack->fThreadId[slot]],
             soaTrack->fEventId[slot], soaTrack->fEkin[slot], lvolID, soaTrack->fStepCounter[slot]);
      survive(LeakStatus::FinishEventOnCPU);
      continue;
    }

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    elTrack.ReSet();
    G4HepEmTrack *theTrack = elTrack.GetTrack();
    theTrack->SetEKin(soaTrack->fEkin[slot]);
    theTrack->SetMCIndex(auxData.fMCIndex);
    theTrack->SetOnBoundary(soaTrack->fNavState[slot].IsOnBoundary());
    theTrack->SetCharge(Charge);
    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    // the default is 1.0e21 but there are float vs double conversions, so we check for 1e20
    mscData->fIsFirstStep        = soaTrack->fInitialRange[slot] > 1.0e+20;
    mscData->fInitialRange       = soaTrack->fInitialRange[slot];
    mscData->fDynamicRangeFactor = soaTrack->fDynamicRangeFactor[slot];
    mscData->fTlimitMin          = soaTrack->fTlimitMin[slot];

    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    // Sample the `number-of-interaction-left` and put it into the track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft = soaTrack->fNumIALeft[slot].values[ip];
      if (numIALeft <= 0) {
        numIALeft = -std::log(soaTrack->Uniform(slot));
      }
      theTrack->SetNumIALeft(numIALeft, ip);
    }

    G4HepEmElectronManager::HowFarToDiscreteInteraction(&g4HepEmData, &g4HepEmPars, &elTrack);

    auto physicalStepLength = elTrack.GetPStepLength();
    // Compute safety, needed for MSC step limit. The accuracy range is physicalStepLength
    double safety = 0.;
    if (!soaTrack->fNavState[slot].IsOnBoundary()) {
      // Get the remaining safety only if larger than physicalStepLength
      safety = soaTrack->GetSafety(slot, soaTrack->fPos[slot]);
      if (safety < physicalStepLength) {
        // Recompute safety and update it in the track.
        // Use maximum accuracy only if safety is samller than physicalStepLength
        safety = AdePTNavigator::ComputeSafety(soaTrack->fPos[slot], soaTrack->fNavState[slot], physicalStepLength);
        soaTrack->SetSafety(slot, soaTrack->fPos[slot], safety);
      }
    }
    theTrack->SetSafety(safety);
    currentTrack.restrictedPhysicalStepLength = false;

    currentTrack.safeLength = 0.;

    if (gMagneticField) {

      const double momentumMag = sqrt(soaTrack->fEkin[slot] * (soaTrack->fEkin[slot] + 2.0 * restMass));
      // Distance along the track direction to reach the maximum allowed error

      // SEVERIN: to be checked if we can use float
      vecgeom::Vector3D<double> momentumVec          = momentumMag * soaTrack->fDir[slot];
      vecgeom::Vector3D<rk_integration_t> B0fieldVec = magneticField.Evaluate(
          soaTrack->fPos[slot][0], soaTrack->fPos[slot][1], soaTrack->fPos[slot][2]); // Field value at starting point
      currentTrack.safeLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, rk_integration_t,
                                    AdePTNavigator>::ComputeSafeLength /*<Real_t>*/ (momentumVec, B0fieldVec, Charge);

      constexpr int MaxSafeLength = 10;
      double limit                = MaxSafeLength * currentTrack.safeLength;
      limit                       = safety > limit ? safety : limit;

      if (physicalStepLength > limit) {
        physicalStepLength                        = limit;
        currentTrack.restrictedPhysicalStepLength = true;
        elTrack.SetPStepLength(physicalStepLength);

        // Note: We are limiting the true step length, which is converted to
        // a shorter geometry step length in HowFarToMSC. In that sense, the
        // limit is an over-approximation, but that is fine for our purpose.
      }
    }

    G4HepEmElectronManager::HowFarToMSC(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Remember MSC values for the next step(s).
    soaTrack->fInitialRange[slot]       = mscData->fInitialRange;
    soaTrack->fDynamicRangeFactor[slot] = mscData->fDynamicRangeFactor;
    soaTrack->fTlimitMin[slot]          = mscData->fTlimitMin;

    // Particles that were not cut or leaked are added to the queue used by the next kernels
    propagationQueue->push_back(slot);
  }
}

template <bool IsElectron>
__global__ void ElectronPropagation(Track *electrons, SoATrack *soaTrack, G4HepEmElectronTrack *hepEMTracks,
                                    const adept::MParray *active, adept::MParray *leakedQueue)
{
  constexpr Precision kPushDistance        = 1000 * vecgeom::kTolerance;
  constexpr int Charge                     = IsElectron ? -1 : 1;
  constexpr double restMass                = copcore::units::kElectronMassC2;
  constexpr int Nvar                       = 6;
  constexpr int max_iterations             = 10;
  constexpr Precision kPushStuck           = 100 * vecgeom::kTolerance;
  constexpr unsigned short kStepsStuckPush = 5;

#ifdef ADEPT_USE_EXT_BFIELD
  using Field_t = GeneralMagneticField;
#else
  using Field_t = UniformMagneticField;
#endif
  using Equation_t = MagneticFieldEquation<Field_t>;
  using Stepper_t  = DormandPrinceRK45<Equation_t, Field_t, Nvar, vecgeom::Precision>;
  using RkDriver_t = RkIntegrationDriver<Stepper_t, vecgeom::Precision, int, Equation_t, Field_t>;

  auto &magneticField = *gMagneticField;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    // Check if there's a volume boundary in between.
    currentTrack.propagated = true;
    currentTrack.hitsurfID  = -1;
    bool zero_first_step    = false;

    if (gMagneticField) {
      int iterDone = -1;
      currentTrack.geometryStepLength =
          fieldPropagatorRungeKutta<Field_t, RkDriver_t, Precision, AdePTNavigator>::ComputeStepAndNextVolume(
              magneticField, soaTrack->fEkin[slot], restMass, Charge, theTrack->GetGStepLength(),
              currentTrack.safeLength, soaTrack->fPos[slot], soaTrack->fDir[slot], soaTrack->fNavState[slot],
              currentTrack.nextState, currentTrack.hitsurfID, currentTrack.propagated,
              /*lengthDone,*/ soaTrack->fSafety[slot],
              // activeSize < 100 ? max_iterations : max_iters_tail ), // Was
              max_iterations, iterDone, slot, zero_first_step);
      // In case of zero step detected by the field propagator this could be due to back scattering, or wrong relocation
      // in the previous step.
      // - In case of BS we should just restore the last exited one for the nextState. For now we cannot detect BS.
      // if (zero_first_step) nextState.SetNavIndex(navState.GetLastExitedState());

    } else {
#ifdef ADEPT_USE_SURF
      currentTrack.geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(
          soaTrack->fPos[slot], soaTrack->fDir[slot], theTrack->GetGStepLength(), soaTrack->fNavState[slot],
          currentTrack.nextState, currentTrack.hitsurfID);
#else
      currentTrack.geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(
          soaTrack->fPos[slot], soaTrack->fDir[slot], theTrack->GetGStepLength(), soaTrack->fNavState[slot],
          currentTrack.nextState, kPushDistance);
#endif
      soaTrack->fPos[slot] += currentTrack.geometryStepLength * soaTrack->fDir[slot];
    }

    if (currentTrack.geometryStepLength < kPushStuck && currentTrack.geometryStepLength < theTrack->GetGStepLength()) {
      soaTrack->fZeroStepCounter[slot]++;
      if (soaTrack->fZeroStepCounter[slot] > kStepsStuckPush) soaTrack->fPos[slot] += kPushStuck * soaTrack->fDir[slot];
    } else
      soaTrack->fZeroStepCounter[slot] = 0;

    // punish miniscule steps by increasing the looperCounter by 10
    if (currentTrack.geometryStepLength < 100 * vecgeom::kTolerance) soaTrack->fLooperCounter[slot] += 10;

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    soaTrack->fNavState[slot].SetBoundaryState(currentTrack.nextState.IsOnBoundary());
    if (currentTrack.nextState.IsOnBoundary()) soaTrack->SetSafety(slot, soaTrack->fPos[slot], 0.);

    // Propagate information from geometrical step to MSC.
    theTrack->SetDirection(soaTrack->fDir[slot].x(), soaTrack->fDir[slot].y(), soaTrack->fDir[slot].z());
    theTrack->SetGStepLength(currentTrack.geometryStepLength);
    theTrack->SetOnBoundary(currentTrack.nextState.IsOnBoundary());
  }
}

template <bool IsElectron>
__global__ void ElectronMSC(Track *electrons, SoATrack *soaTrack, G4HepEmElectronTrack *hepEMTracks,
                            const adept::MParray *active)
{
  constexpr double restMass = copcore::units::kElectronMassC2;

  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    // Apply continuous effects.
    currentTrack.stopped = G4HepEmElectronManager::PerformContinuous(&g4HepEmData, &g4HepEmPars, &elTrack, &rnge);

    // Collect the direction change and displacement by MSC.
    const double *direction = theTrack->GetDirection();
    soaTrack->fDir[slot].Set(direction[0], direction[1], direction[2]);
    if (!currentTrack.nextState.IsOnBoundary()) {
      const double *mscDisplacement = mscData->GetDisplacement();
      vecgeom::Vector3D<Precision> displacement(mscDisplacement[0], mscDisplacement[1], mscDisplacement[2]);
      const double dLength2            = displacement.Length2();
      constexpr double kGeomMinLength  = 5 * copcore::units::nm;          // 0.05 [nm]
      constexpr double kGeomMinLength2 = kGeomMinLength * kGeomMinLength; // (0.05 [nm])^2
      if (dLength2 > kGeomMinLength2) {
        const double dispR = std::sqrt(dLength2);
        // Estimate safety by subtracting the geometrical step length.
        double safety          = soaTrack->GetSafety(slot, soaTrack->fPos[slot]);
        constexpr double sFact = 0.99;
        double reducedSafety   = sFact * safety;

        // Apply displacement, depending on how close we are to a boundary.
        // 1a. Far away from geometry boundary:
        if (reducedSafety > 0.0 && dispR <= reducedSafety) {
          soaTrack->fPos[slot] += displacement;
        } else {
          // Recompute safety.
          // Use maximum accuracy only if safety is samller than physicalStepLength
          safety = AdePTNavigator::ComputeSafety(soaTrack->fPos[slot], soaTrack->fNavState[slot], dispR);
          soaTrack->SetSafety(slot, soaTrack->fPos[slot], safety);
          reducedSafety = sFact * safety;

          // 1b. Far away from geometry boundary:
          if (reducedSafety > 0.0 && dispR <= reducedSafety) {
            soaTrack->fPos[slot] += displacement;
            // 2. Push to boundary:
          } else if (reducedSafety > kGeomMinLength) {
            soaTrack->fPos[slot] += displacement * (reducedSafety / dispR);
          }
          // 3. Very small safety: do nothing.
        }
      }
    }

    // Collect the charged step length (might be changed by MSC). Collect the changes in energy and deposit.
    soaTrack->fEkin[slot] = theTrack->GetEKin();

    // Update the flight times of the particle
    // By calculating the velocity here, we assume that all the energy deposit is done at the PreStepPoint, and
    // the velocity depends on the remaining energy
    double deltaTime = elTrack.GetPStepLength() / GetVelocity(soaTrack->fEkin[slot]);
    soaTrack->fGlobalTime[slot] += deltaTime;
    soaTrack->fLocalTime[slot] += deltaTime;
    soaTrack->fProperTime[slot] += deltaTime * (restMass / soaTrack->fEkin[slot]);
  }
}

/***
 * @brief Adds tracks to interaction and relocation queues depending on their state
 */
template <bool IsElectron, typename Scoring>
__global__ void ElectronSetupInteractions(Track *electrons, SoATrack *soaTrack, Track *leaks, SoATrack *soaLeaks,
                                          G4HepEmElectronTrack *hepEMTracks, const adept::MParray *active,
                                          Secondaries secondaries, adept::MParray *nextActiveQueue,
                                          AllInteractionQueues interactionQueues, adept::MParray *leakedQueue,
                                          Scoring *userScoring, const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*active)[i];
    // SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager :
    // *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      soaTrack->fLeakStatus[slot] = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot;
        if (IsElectron)
          leakSlot = secondaries.electrons.NextLeakSlot();
        else
          leakSlot = secondaries.positrons.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = electrons[slot];
        CopyTrack(slot, leakSlot, electrons, leaks, soaTrack, soaLeaks);
        auto success = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        // slotManager.MarkSlotForFreeing(slot);
      } else {
        // Get a slot in the next active queue
        // This is necessary as we are copying the track to the next array
        unsigned int nextSlot{0};
        if (IsElectron) {
          nextSlot = secondaries.electrons.NextSlot();
          // Copy the track to the next active tracks array
          CopyTrack(slot, nextSlot, electrons, secondaries.electrons.fNextTracks, soaTrack,
                    secondaries.electrons.fSoANextTracks);
        } else {
          nextSlot = secondaries.positrons.NextSlot();
          // Copy the track to the next active tracks array
          CopyTrack(slot, nextSlot, electrons, secondaries.positrons.fNextTracks, soaTrack,
                    secondaries.positrons.fSoANextTracks);
        }
        // Store the slot in the next active queue
        nextActiveQueue->push_back(nextSlot);
      }
    };

    bool isLastStep = false;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool reached_interaction = true;
    bool printErrors         = false;

    // Save the `number-of-interaction-left` in our track.
    for (int ip = 0; ip < 4; ++ip) {
      double numIALeft                      = theTrack->GetNumIALeft(ip);
      soaTrack->fNumIALeft[slot].values[ip] = numIALeft;
    }

    // Set Non-stopped, on-boundary tracks for relocation
    if (currentTrack.nextState.IsOnBoundary() && !currentTrack.stopped) {
      // Add particle to relocation queue
      interactionQueues.queues[4]->push_back(slot);
      continue;
    }

    // Now check whether the non-relocating tracks reached an interaction
    if (!currentTrack.stopped) {
      if (!currentTrack.propagated || currentTrack.restrictedPhysicalStepLength) {
        // Did not yet reach the interaction point due to error in the magnetic
        // field propagation. Try again next time.

        if (++soaTrack->fLooperCounter[slot] > 500) {
          // Kill loopers that are not advancing in free space
          if (printErrors)
            printf("Killing looper due to lack of advance: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
                   "physicsStepLength=%E "
                   "safety=%E\n",
                   soaTrack->fEkin[slot], soaTrack->fEventId[slot], soaTrack->fLooperCounter[slot], energyDeposit,
                   currentTrack.geometryStepLength, theTrack->GetGStepLength(), soaTrack->fSafety[slot]);
          continue;
        }

        // mark winner process to be transport, although this is not strictly true
        theTrack->SetWinnerProcessIndex(10);

        survive();
        reached_interaction = false;
      } else if (theTrack->GetWinnerProcessIndex() < 0) {
        // No discrete process, move on.
        survive();
        reached_interaction = false;
      } else if (G4HepEmElectronManager::CheckDelta(&g4HepEmData, theTrack, soaTrack->Uniform(slot))) {
        // If there was a delta interaction, the track survives but does not move onto the next kernel
        survive();
        reached_interaction = false;
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        soaTrack->fNumIALeft[slot].values[theTrack->GetWinnerProcessIndex()] = -1.0;
      }
    } else {
      // Stopped positrons annihilate, stopped electrons score and die
      if (IsElectron) {
        reached_interaction = false;
        isLastStep          = true;
        // Ekin = 0 for correct scoring
        soaTrack->fEkin[slot] = 0;
        // Particle is killed by not enqueuing it for the next iteration. Free the slot it occupies
        // slotManager.MarkSlotForFreeing(slot);
      }
    }

    // Now push the particles that reached their interaction into the per-interaction queues
    if (reached_interaction) {
      // reset Looper counter if limited by discrete interaction or MSC
      soaTrack->fLooperCounter[slot] = 0;

      if (!currentTrack.stopped) {
        // Reset number of interaction left for the winner discrete process.
        // (Will be resampled in the next iteration.)
        soaTrack->fNumIALeft[slot].values[theTrack->GetWinnerProcessIndex()] = -1.0;
        // Enqueue the particles
        if (theTrack->GetWinnerProcessIndex() < 3) {
          interactionQueues.queues[theTrack->GetWinnerProcessIndex()]->push_back(slot);
        } else {
          // Lepton nuclear needs to be handled by Geant4 directly, passing track back to CPU
          survive(LeakStatus::LeptonNuclear);
        }
      } else {
        // Stopped positron
        interactionQueues.queues[3]->push_back(slot);
      }

    } else {
      // Only non-interacting, non-relocating tracks score here
      // Note: In this kernel returnLastStep is only true for particles that left the world
      // Score the edep for particles that didn't reach the interaction
      if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
        adept_scoring::RecordHit(userScoring,
                                 soaTrack->fTrackId[slot],                              // Track ID
                                 soaTrack->fParentId[slot],                             // parent Track ID
                                 static_cast<short>(theTrack->GetWinnerProcessIndex()), // step defining process
                                 static_cast<char>(IsElectron ? 0 : 1),                 // Particle type
                                 elTrack.GetPStepLength(),                              // Step length
                                 energyDeposit,                                         // Total Edep
                                 soaTrack->fWeight[slot],                               // Track weight
                                 soaTrack->fNavState[slot],                             // Pre-step point navstate
                                 currentTrack.preStepPos,                               // Pre-step point position
                                 currentTrack.preStepDir,     // Pre-step point momentum direction
                                 currentTrack.preStepEKin,    // Pre-step point kinetic energy
                                 currentTrack.nextState,      // Post-step point navstate
                                 soaTrack->fPos[slot],        // Post-step point position
                                 soaTrack->fDir[slot],        // Post-step point momentum direction
                                 soaTrack->fEkin[slot],       // Post-step point kinetic energy
                                 soaTrack->fGlobalTime[slot], // global time
                                 soaTrack->fLocalTime[slot],  // local time
                                 soaTrack->fEventId[slot], soaTrack->fThreadId[slot], // eventID and threadID
                                 isLastStep,                                          // whether this was the last step
                                 soaTrack->fStepCounter[slot]);                       // stepcounter
    }
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronRelocation(Track *electrons, SoATrack *soaTrack, Track *leaks, SoATrack *soaLeaks,
                                   G4HepEmElectronTrack *hepEMTracks, Secondaries secondaries,
                                   adept::MParray *nextActiveQueue, adept::MParray *relocatingQueue,
                                   adept::MParray *leakedQueue, Scoring *userScoring, const bool returnAllSteps,
                                   const bool returnLastStep)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  int activeSize                    = relocatingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*relocatingQueue)[i];
    // SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager :
    // *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData

    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      // NOTE: When adapting the split kernels for async mode this won't
      // work if we want to re-use slots on the fly. Directly copying to
      // a trackdata struct would be better
      soaTrack->fLeakStatus[slot] = leakReason;
      if (leakReason != LeakStatus::NoLeak) {
        // Get a slot in the leaks array
        int leakSlot;
        if (IsElectron)
          leakSlot = secondaries.electrons.NextLeakSlot();
        else
          leakSlot = secondaries.positrons.NextLeakSlot();
        // Copy the track to the leaks array and store the index in the leak queue
        leaks[leakSlot] = electrons[slot];
        CopyTrack(slot, leakSlot, electrons, leaks, soaTrack, soaLeaks);
        auto success = leakedQueue->push_back(leakSlot);
        if (!success) {
          printf("ERROR: No space left in e-/+ leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
        // Free the slot in the tracks slot manager
        // slotManager.MarkSlotForFreeing(slot);
      } else {
        // Get a slot in the next active queue
        // This is necessary as we are copying the track to the next array
        unsigned int nextSlot{0};
        if (IsElectron) {
          nextSlot = secondaries.electrons.NextSlot();
          // Copy the track to the next active tracks array
          CopyTrack(slot, nextSlot, electrons, secondaries.electrons.fNextTracks, soaTrack,
                    secondaries.electrons.fSoANextTracks);
        } else {
          nextSlot = secondaries.positrons.NextSlot();
          // Copy the track to the next active tracks array
          CopyTrack(slot, nextSlot, electrons, secondaries.positrons.fNextTracks, soaTrack,
                    secondaries.positrons.fSoANextTracks);
        }
        // Store the slot in the next active queue
        nextActiveQueue->push_back(nextSlot);
      }
    };

    bool isLastStep = false;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    double energyDeposit = theTrack->GetEnergyDeposit();

    bool cross_boundary = false;
    bool printErrors    = false;

    // Relocate to have the correct next state before RecordHit is called

    // - Kill loopers stuck at a boundary
    // - Set cross boundary flag in order to set the correct navstate after scoring
    // - Kill particles that left the world

    if (++soaTrack->fLooperCounter[slot] > 500) {
      // Kill loopers that are scraping a boundary
      if (printErrors)
        printf("Killing looper scraping at a boundary: E=%E event=%d loop=%d energyDeposit=%E geoStepLength=%E "
               "physicsStepLength=%E "
               "safety=%E\n",
               soaTrack->fEkin[slot], soaTrack->fEventId[slot], soaTrack->fLooperCounter[slot], energyDeposit,
               currentTrack.geometryStepLength, theTrack->GetGStepLength(), soaTrack->fSafety[slot]);
      continue;
    }

    if (!currentTrack.nextState.IsOutside()) {
      // Mark the particle. We need to change its navigation state to the next volume before enqueuing it
      // This will happen after recording the step
      // Relocate
      cross_boundary = true;
#ifdef ADEPT_USE_SURF
      AdePTNavigator::RelocateToNextVolume(soaTrack->fPos[slot], soaTrack->fDir[slot], currentTrack.hitsurfID,
                                           currentTrack.nextState);
#else
      AdePTNavigator::RelocateToNextVolume(soaTrack->fPos[slot], soaTrack->fDir[slot], currentTrack.nextState);
#endif
      // Set the last exited state to be the one before crossing
      currentTrack.nextState.SetLastExited(soaTrack->fNavState[slot].GetState());
    } else {
      // Particle left the world, don't enqueue it and release the slot
      // slotManager.MarkSlotForFreeing(slot);
      isLastStep = true;
    }

    // Score
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring,
                               soaTrack->fTrackId[slot],              // Track ID
                               soaTrack->fParentId[slot],             // parent Track ID
                               static_cast<short>(/*transport*/ 10),  // step limiting process ID
                               static_cast<char>(IsElectron ? 0 : 1), // Particle type
                               elTrack.GetPStepLength(),              // Step length
                               energyDeposit,                         // Total Edep
                               soaTrack->fWeight[slot],               // Track weight
                               soaTrack->fNavState[slot],             // Pre-step point navstate
                               currentTrack.preStepPos,               // Pre-step point position
                               currentTrack.preStepDir,               // Pre-step point momentum direction
                               currentTrack.preStepEKin,              // Pre-step point kinetic energy
                               currentTrack.nextState,                // Post-step point navstate
                               soaTrack->fPos[slot],                  // Post-step point position
                               soaTrack->fDir[slot],                  // Post-step point momentum direction
                               soaTrack->fEkin[slot],                 // Post-step point kinetic energy
                               soaTrack->fGlobalTime[slot],           // global time
                               soaTrack->fLocalTime[slot],            // local time
                               soaTrack->fEventId[slot], soaTrack->fThreadId[slot], // eventID and threadID
                               isLastStep,                                          // whether this was the last step
                               soaTrack->fStepCounter[slot]);                       // stepcounter

    if (cross_boundary) {
      // Move to the next boundary now that the Step is recorded
      soaTrack->fNavState[slot] = currentTrack.nextState;
      // Check if the next volume belongs to the GPU region and push it to the appropriate queue
      const int nextlvolID          = soaTrack->fNavState[slot].GetLogicalId();
      VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
      if (nextauxData.fGPUregion > 0) {
        survive();
      } else {
        // To be safe, just push a bit the track exiting the GPU region to make sure
        // Geant4 does not relocate it again inside the same region
        soaTrack->fPos[slot] += kPushDistance * soaTrack->fDir[slot];
        survive(LeakStatus::OutOfGPURegion);
      }
    }
  }
}

template <bool IsElectron, typename Scoring>
__device__ __forceinline__ void PerformStoppedAnnihilation(const int slot, Track &currentTrack, SoATrack *soaTrack,
                                                           Secondaries &secondaries, double &energyDeposit,
                                                           SlotManager &slotManager, const bool ApplyCuts,
                                                           const double theGammaCut, Scoring *userScoring,
                                                           const bool returnLastStep = false)
{
  soaTrack->fEkin[slot] = 0;
  if (!IsElectron) {
    // Annihilate the stopped positron into two gammas heading to opposite
    // directions (isotropic).

    // Apply cuts
    if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut)) {
      // Deposit the energy here and don't initialize any secondaries
      energyDeposit += 2 * copcore::units::kElectronMassC2;
    } else {

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

      const double cost = 2 * soaTrack->Uniform(slot) - 1;
      const double sint = sqrt(1 - cost * cost);
      const double phi  = k2Pi * soaTrack->Uniform(slot);
      double sinPhi, cosPhi;
      sincos(phi, &sinPhi, &cosPhi);

      // as the other branched newRNG may have already been used by interactions before, we need to advance and create a
      // new one
      soaTrack->fRngState[slot].Advance();
      RanluxppDouble newRNG(soaTrack->fRngState[slot].Branch());

      Track &gamma1 = secondaries.gammas.NextTrack(double{copcore::units::kElectronMassC2}, soaTrack->fPos[slot],
                                                   vecgeom::Vector3D<Precision>{sint * cosPhi, sint * sinPhi, cost},
                                                   soaTrack->fGlobalTime[slot], soaTrack, slot, newRNG,
                                                   soaTrack->fNavState[slot], currentTrack);

      // Reuse the RNG state of the dying track.
      Track &gamma2 = secondaries.gammas.NextTrack(double{copcore::units::kElectronMassC2}, soaTrack->fPos[slot],
                                                   -secondaries.gammas.fSoANextTracks->fDir[gamma1.currentSlot],
                                                   soaTrack->fGlobalTime[slot], soaTrack, slot,
                                                   soaTrack->fRngState[slot], soaTrack->fNavState[slot], currentTrack);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(
            userScoring, secondaries.gammas.fSoANextTracks->fTrackId[gamma1.currentSlot],
            secondaries.gammas.fSoANextTracks->fParentId[gamma1.currentSlot],
            /*CreatorProcessId*/ short(2),
            /* gamma*/ 2,                                                       // Particle type
            0,                                                                  // Step length
            0,                                                                  // Total Edep
            secondaries.gammas.fSoANextTracks->fWeight[gamma1.currentSlot],     // Track weight
            secondaries.gammas.fSoANextTracks->fNavState[gamma1.currentSlot],   // Pre-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma1.currentSlot],        // Pre-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma1.currentSlot],        // Pre-step point momentum direction
            double{copcore::units::kElectronMassC2},                            // Pre-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fNavState[gamma1.currentSlot],   // Post-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma1.currentSlot],        // Post-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma1.currentSlot],        // Post-step point momentum direction
            double{copcore::units::kElectronMassC2},                            // Post-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fGlobalTime[gamma1.currentSlot], // global time
            0.,                                                                 // local time
            secondaries.gammas.fSoANextTracks->fEventId[gamma1.currentSlot],
            secondaries.gammas.fSoANextTracks->fThreadId[gamma1.currentSlot],     // eventID and threadID
            false,                                                                // whether this was the last step
            secondaries.gammas.fSoANextTracks->fStepCounter[gamma1.currentSlot]); // whether this was the first step
        adept_scoring::RecordHit(
            userScoring, secondaries.gammas.fSoANextTracks->fTrackId[gamma2.currentSlot],
            secondaries.gammas.fSoANextTracks->fParentId[gamma2.currentSlot],
            /*CreatorProcessId*/ short(2),
            /* gamma*/ 2,                                                       // Particle type
            0,                                                                  // Step length
            0,                                                                  // Total Edep
            secondaries.gammas.fSoANextTracks->fWeight[gamma2.currentSlot],     // Track weight
            secondaries.gammas.fSoANextTracks->fNavState[gamma2.currentSlot],   // Pre-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma2.currentSlot],        // Pre-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma2.currentSlot],        // Pre-step point momentum direction
            double{copcore::units::kElectronMassC2},                            // Pre-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fNavState[gamma2.currentSlot],   // Post-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma2.currentSlot],        // Post-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma2.currentSlot],        // Post-step point momentum direction
            double{copcore::units::kElectronMassC2},                            // Post-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fGlobalTime[gamma2.currentSlot], // global time
            0.,                                                                 // local time
            secondaries.gammas.fSoANextTracks->fEventId[gamma2.currentSlot],
            secondaries.gammas.fSoANextTracks->fThreadId[gamma2.currentSlot],     // eventID and threadID
            false,                                                                // whether this was the last step
            secondaries.gammas.fSoANextTracks->fStepCounter[gamma2.currentSlot]); // whether this was the first step
      }
    }
  }
  // Particles are killed by not enqueuing them into the new activeQueue (and free the slot in async mode)
  // slotManager.MarkSlotForFreeing(slot);
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronIonization(Track *electrons, SoATrack *soaTrack, G4HepEmElectronTrack *hepEMTracks,
                                   Secondaries secondaries, adept::MParray *nextActiveQueue,
                                   adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                   const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    auto survive = [&]() {
      isLastStep = false; // track survived, do not force return of step
      // Get a slot in the next active queue
      // This is necessary as we are copying the track to the next array
      unsigned int nextSlot{0};
      if (IsElectron) {
        nextSlot = secondaries.electrons.NextSlot();
        // Copy the track to the next active tracks array
        CopyTrack(slot, nextSlot, electrons, secondaries.electrons.fNextTracks, soaTrack,
                  secondaries.electrons.fSoANextTracks);
      } else {
        nextSlot = secondaries.positrons.NextSlot();
        // Copy the track to the next active tracks array
        CopyTrack(slot, nextSlot, electrons, secondaries.positrons.fNextTracks, soaTrack,
                  secondaries.positrons.fSoANextTracks);
      }
      // Store the slot in the next active queue
      nextActiveQueue->push_back(nextSlot);
    };

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Perform the discrete interaction, branch a new RNG state with advance so it is
    // ready to be used.
    auto newRNG = RanluxppDouble(soaTrack->fRngState[slot].Branch());
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    soaTrack->fRngState[slot].Advance();

    // Invoke ionization (for e-/e+):
    double deltaEkin =
        (IsElectron) ? G4HepEmElectronInteractionIoni::SampleETransferMoller(theElCut, soaTrack->fEkin[slot], &rnge)
                     : G4HepEmElectronInteractionIoni::SampleETransferBhabha(theElCut, soaTrack->fEkin[slot], &rnge);

    double dirPrimary[] = {soaTrack->fDir[slot].x(), soaTrack->fDir[slot].y(), soaTrack->fDir[slot].z()};
    double dirSecondary[3];
    G4HepEmElectronInteractionIoni::SampleDirections(soaTrack->fEkin[slot], deltaEkin, dirSecondary, dirPrimary, &rnge);

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theElCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      Track &secondary = secondaries.electrons.NextTrack(
          deltaEkin, soaTrack->fPos[slot],
          vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, soaTrack->fGlobalTime[slot],
          soaTrack, slot, newRNG, soaTrack->fNavState[slot], currentTrack);

      // printf("REAL dir: (%f, %f, %f)\n", dirSecondary[0], dirSecondary[1], dirSecondary[2]);
      // printf("secondary dir: (%f, %f, %f), SLOT: %d, SOA: %p, SOA DIR: %p, SOA NEXT: %p, SOA NEXT DIR: %p\n",
      //        soaNextTrack->fDir[secondary.currentSlot].x(), soaNextTrack->fDir[secondary.currentSlot].y(),
      //        soaNextTrack->fDir[secondary.currentSlot].z(), secondary.currentSlot, soaTrack, soaTrack->fDir,
      //        soaNextTrack, soaNextTrack->fDir);

      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(
            userScoring, secondaries.electrons.fSoANextTracks->fTrackId[secondary.currentSlot],
            secondaries.electrons.fSoANextTracks->fParentId[secondary.currentSlot],
            /*CreatorProcessId*/ short(0),
            /* electron*/ 0,                                                        // Particle type
            0,                                                                      // Step length
            0,                                                                      // Total Edep
            secondaries.electrons.fSoANextTracks->fWeight[secondary.currentSlot],   // Track weight
            secondaries.electrons.fSoANextTracks->fNavState[secondary.currentSlot], // Pre-step point navstate
            secondaries.electrons.fSoANextTracks->fPos[secondary.currentSlot],      // Pre-step point position
            secondaries.electrons.fSoANextTracks->fDir[secondary.currentSlot],      // Pre-step point momentum direction
            deltaEkin,                                                              // Pre-step point kinetic energy
            secondaries.electrons.fSoANextTracks->fNavState[secondary.currentSlot], // Post-step point navstate
            secondaries.electrons.fSoANextTracks->fPos[secondary.currentSlot],      // Post-step point position
            secondaries.electrons.fSoANextTracks->fDir[secondary.currentSlot], // Post-step point momentum direction
            deltaEkin,                                                         // Post-step point kinetic energy
            secondaries.electrons.fSoANextTracks->fGlobalTime[secondary.currentSlot], // global time
            0.,                                                                       // local time
            secondaries.electrons.fSoANextTracks->fEventId[secondary.currentSlot],
            secondaries.electrons.fSoANextTracks->fThreadId[secondary.currentSlot], // eventID and threadID
            false,                                                                  // whether this was the last step
            secondaries.electrons.fSoANextTracks
                ->fStepCounter[secondary.currentSlot]); // whether this was the first step
      }
    }

    soaTrack->fEkin[slot] -= deltaEkin;

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (soaTrack->fEkin[slot] < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += soaTrack->fEkin[slot];
      }
      currentTrack.stopped = true;
      PerformStoppedAnnihilation<IsElectron, Scoring>(slot, currentTrack, soaTrack, secondaries, energyDeposit,
                                                      slotManager, ApplyCuts, theGammaCut, userScoring, returnLastStep);
    } else {
      soaTrack->fDir[slot].Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring,
                               soaTrack->fTrackId[slot],              // Track ID
                               soaTrack->fParentId[slot],             // parent Track ID
                               static_cast<short>(0),                 // step limiting process ID
                               static_cast<char>(IsElectron ? 0 : 1), // Particle type
                               elTrack.GetPStepLength(),              // Step length
                               energyDeposit,                         // Total Edep
                               soaTrack->fWeight[slot],               // Track weight
                               soaTrack->fNavState[slot],             // Pre-step point navstate
                               currentTrack.preStepPos,               // Pre-step point position
                               currentTrack.preStepDir,               // Pre-step point momentum direction
                               currentTrack.preStepEKin,              // Pre-step point kinetic energy
                               currentTrack.nextState,                // Post-step point navstate
                               soaTrack->fPos[slot],                  // Post-step point position
                               soaTrack->fDir[slot],                  // Post-step point momentum direction
                               soaTrack->fEkin[slot],                 // Post-step point kinetic energy
                               soaTrack->fGlobalTime[slot],           // global time
                               soaTrack->fLocalTime[slot],            // local time
                               soaTrack->fEventId[slot], soaTrack->fThreadId[slot], // eventID and threadID
                               isLastStep,                                          // whether this was the last step
                               soaTrack->fStepCounter[slot]);                       // stepcounter
  }
}

template <bool IsElectron, typename Scoring>
__global__ void ElectronBremsstrahlung(Track *electrons, SoATrack *soaTrack, G4HepEmElectronTrack *hepEMTracks,
                                       Secondaries secondaries, adept::MParray *nextActiveQueue,
                                       adept::MParray *interactingQueue, Scoring *userScoring,
                                       const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = IsElectron ? *secondaries.electrons.fSlotManager : *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    auto survive = [&]() {
      isLastStep = false; // track survived, do not force return of step
      // Get a slot in the next active queue
      // This is necessary as we are copying the track to the next array
      unsigned int nextSlot{0};
      if (IsElectron) {
        nextSlot = secondaries.electrons.NextSlot();
        // Copy the track to the next active tracks array
        CopyTrack(slot, nextSlot, electrons, secondaries.electrons.fNextTracks, soaTrack,
                  secondaries.electrons.fSoANextTracks);
      } else {
        nextSlot = secondaries.positrons.NextSlot();
        // Copy the track to the next active tracks array
        CopyTrack(slot, nextSlot, electrons, secondaries.positrons.fNextTracks, soaTrack,
                  secondaries.positrons.fSoANextTracks);
      }
      // Store the slot in the next active queue
      nextActiveQueue->push_back(nextSlot);
    };

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Perform the discrete interaction, branch a new RNG state with advance so it is
    // ready to be used.
    auto newRNG = RanluxppDouble(soaTrack->fRngState[slot].Branch());
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    soaTrack->fRngState[slot].Advance();

    // Invoke model for Bremsstrahlung: either SB- or Rel-Brem.
    double logEnergy = std::log(soaTrack->fEkin[slot]);
    double deltaEkin = soaTrack->fEkin[slot] < g4HepEmPars.fElectronBremModelLim
                           ? G4HepEmElectronInteractionBrem::SampleETransferSB(
                                 &g4HepEmData, soaTrack->fEkin[slot], logEnergy, auxData.fMCIndex, &rnge, IsElectron)
                           : G4HepEmElectronInteractionBrem::SampleETransferRB(
                                 &g4HepEmData, soaTrack->fEkin[slot], logEnergy, auxData.fMCIndex, &rnge, IsElectron);

    double dirPrimary[] = {soaTrack->fDir[slot].x(), soaTrack->fDir[slot].y(), soaTrack->fDir[slot].z()};
    double dirSecondary[3];
    G4HepEmElectronInteractionBrem::SampleDirections(soaTrack->fEkin[slot], deltaEkin, dirSecondary, dirPrimary, &rnge);

    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 1);

    // Apply cuts
    if (ApplyCuts && (deltaEkin < theGammaCut)) {
      // Deposit the energy here and kill the secondary
      energyDeposit += deltaEkin;

    } else {
      Track &gamma = secondaries.gammas.NextTrack(
          deltaEkin, soaTrack->fPos[slot],
          vecgeom::Vector3D<Precision>{dirSecondary[0], dirSecondary[1], dirSecondary[2]}, soaTrack->fGlobalTime[slot],
          soaTrack, slot, newRNG, soaTrack->fNavState[slot], currentTrack);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(
            userScoring, secondaries.gammas.fSoANextTracks->fTrackId[gamma.currentSlot],
            secondaries.gammas.fSoANextTracks->fParentId[gamma.currentSlot],
            /*CreatorProcessId*/ short(1),
            /* gamma*/ 2,                                                      // Particle type
            0,                                                                 // Step length
            0,                                                                 // Total Edep
            secondaries.gammas.fSoANextTracks->fWeight[gamma.currentSlot],     // Track weight
            secondaries.gammas.fSoANextTracks->fNavState[gamma.currentSlot],   // Pre-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma.currentSlot],        // Pre-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma.currentSlot],        // Pre-step point momentum direction
            deltaEkin,                                                         // Pre-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fNavState[gamma.currentSlot],   // Post-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma.currentSlot],        // Post-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma.currentSlot],        // Post-step point momentum direction
            deltaEkin,                                                         // Post-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fGlobalTime[gamma.currentSlot], // global time
            0.,                                                                // local time
            secondaries.gammas.fSoANextTracks->fEventId[gamma.currentSlot],
            secondaries.gammas.fSoANextTracks->fThreadId[gamma.currentSlot],     // eventID and threadID
            false,                                                               // whether this was the last step
            secondaries.gammas.fSoANextTracks->fStepCounter[gamma.currentSlot]); // whether this was the first step
      }
    }

    soaTrack->fEkin[slot] -= deltaEkin;

    // if below tracking cut, deposit energy for electrons (positrons are annihilated later) and stop particles
    if (soaTrack->fEkin[slot] < g4HepEmPars.fElectronTrackingCut) {
      if (IsElectron) {
        energyDeposit += soaTrack->fEkin[slot];
      }
      currentTrack.stopped = true;
      PerformStoppedAnnihilation<IsElectron, Scoring>(slot, currentTrack, soaTrack, secondaries, energyDeposit,
                                                      slotManager, ApplyCuts, theGammaCut, userScoring, returnLastStep);
    } else {
      soaTrack->fDir[slot].Set(dirPrimary[0], dirPrimary[1], dirPrimary[2]);
      survive();
    }

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring,
                               soaTrack->fTrackId[slot],              // Track ID
                               soaTrack->fParentId[slot],             // parent Track ID
                               static_cast<short>(1),                 // step limiting process ID
                               static_cast<char>(IsElectron ? 0 : 1), // Particle type
                               elTrack.GetPStepLength(),              // Step length
                               energyDeposit,                         // Total Edep
                               soaTrack->fWeight[slot],               // Track weight
                               soaTrack->fNavState[slot],             // Pre-step point navstate
                               currentTrack.preStepPos,               // Pre-step point position
                               currentTrack.preStepDir,               // Pre-step point momentum direction
                               currentTrack.preStepEKin,              // Pre-step point kinetic energy
                               currentTrack.nextState,                // Post-step point navstate
                               soaTrack->fPos[slot],                  // Post-step point position
                               soaTrack->fDir[slot],                  // Post-step point momentum direction
                               soaTrack->fEkin[slot],                 // Post-step point kinetic energy
                               soaTrack->fGlobalTime[slot],           // global time
                               soaTrack->fLocalTime[slot],            // local time
                               soaTrack->fEventId[slot], soaTrack->fThreadId[slot], // eventID and threadID
                               isLastStep,                                          // whether this was the last step
                               soaTrack->fStepCounter[slot]);                       // stepcounter
  }
}

template <typename Scoring>
__global__ void PositronAnnihilation(Track *electrons, SoATrack *soaTrack, G4HepEmElectronTrack *hepEMTracks,
                                     Secondaries secondaries, adept::MParray *nextActiveQueue,
                                     adept::MParray *interactingQueue, Scoring *userScoring, const bool returnAllSteps,
                                     const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot = (*interactingQueue)[i];
    // SlotManager &slotManager = *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Perform the discrete interaction, branch a new RNG state with advance so it is
    // ready to be used.
    auto newRNG = RanluxppDouble(soaTrack->fRngState[slot].Branch());
    // Also advance the current RNG state to provide a fresh round of random
    // numbers after MSC used up a fair share for sampling the displacement.
    soaTrack->fRngState[slot].Advance();

    // Invoke annihilation (in-flight) for e+
    double dirPrimary[] = {soaTrack->fDir[slot].x(), soaTrack->fDir[slot].y(), soaTrack->fDir[slot].z()};
    double theGamma1Ekin, theGamma2Ekin;
    double theGamma1Dir[3], theGamma2Dir[3];
    G4HepEmPositronInteractionAnnihilation::SampleEnergyAndDirectionsInFlight(
        soaTrack->fEkin[slot], dirPrimary, &theGamma1Ekin, theGamma1Dir, &theGamma2Ekin, theGamma2Dir, &rnge);

    // TODO: In principle particles are produced, then cut before stacking them. It seems correct to count them
    // here
    adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 0, /*numPositrons*/ 0, /*numGammas*/ 2);

    // Apply cuts
    if (ApplyCuts && (theGamma1Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma1Ekin;

    } else {
      Track &gamma1 = secondaries.gammas.NextTrack(
          theGamma1Ekin, soaTrack->fPos[slot],
          vecgeom::Vector3D<Precision>{theGamma1Dir[0], theGamma1Dir[1], theGamma1Dir[2]}, soaTrack->fGlobalTime[slot],
          soaTrack, slot, newRNG, soaTrack->fNavState[slot], currentTrack);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(
            userScoring, secondaries.gammas.fSoANextTracks->fTrackId[gamma1.currentSlot],
            secondaries.gammas.fSoANextTracks->fParentId[gamma1.currentSlot],
            /*CreatorProcessId*/ short(2),
            /* gamma*/ 2,                                                       // Particle type
            0,                                                                  // Step length
            0,                                                                  // Total Edep
            secondaries.gammas.fSoANextTracks->fWeight[gamma1.currentSlot],     // Track weight
            secondaries.gammas.fSoANextTracks->fNavState[gamma1.currentSlot],   // Pre-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma1.currentSlot],        // Pre-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma1.currentSlot],        // Pre-step point momentum direction
            theGamma1Ekin,                                                      // Pre-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fNavState[gamma1.currentSlot],   // Post-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma1.currentSlot],        // Post-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma1.currentSlot],        // Post-step point momentum direction
            theGamma1Ekin,                                                      // Post-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fGlobalTime[gamma1.currentSlot], // global time
            0.,                                                                 // local time
            secondaries.gammas.fSoANextTracks->fEventId[gamma1.currentSlot],
            secondaries.gammas.fSoANextTracks->fThreadId[gamma1.currentSlot], // eventID and threadID
            false,                                                            // whether this was the last step
            secondaries.gammas.fSoANextTracks->fStepCounter[gamma1.currentSlot]);
      }
    }
    if (ApplyCuts && (theGamma2Ekin < theGammaCut)) {
      // Deposit the energy here and kill the secondaries
      energyDeposit += theGamma2Ekin;

    } else {
      Track &gamma2 = secondaries.gammas.NextTrack(
          theGamma2Ekin, soaTrack->fPos[slot],
          vecgeom::Vector3D<Precision>{theGamma2Dir[0], theGamma2Dir[1], theGamma2Dir[2]}, soaTrack->fGlobalTime[slot],
          soaTrack, slot, soaTrack->fRngState[slot], soaTrack->fNavState[slot], currentTrack);
      // if tracking or stepping action is called, return initial step
      if (returnLastStep) {
        adept_scoring::RecordHit(
            userScoring, secondaries.gammas.fSoANextTracks->fTrackId[gamma2.currentSlot],
            secondaries.gammas.fSoANextTracks->fParentId[gamma2.currentSlot],
            /*CreatorProcessId*/ short(2),
            /* gamma*/ 2,                                                       // Particle type
            0,                                                                  // Step length
            0,                                                                  // Total Edep
            secondaries.gammas.fSoANextTracks->fWeight[gamma2.currentSlot],     // Track weight
            secondaries.gammas.fSoANextTracks->fNavState[gamma2.currentSlot],   // Pre-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma2.currentSlot],        // Pre-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma2.currentSlot],        // Pre-step point momentum direction
            theGamma2Ekin,                                                      // Pre-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fNavState[gamma2.currentSlot],   // Post-step point navstate
            secondaries.gammas.fSoANextTracks->fPos[gamma2.currentSlot],        // Post-step point position
            secondaries.gammas.fSoANextTracks->fDir[gamma2.currentSlot],        // Post-step point momentum direction
            theGamma2Ekin,                                                      // Post-step point kinetic energy
            secondaries.gammas.fSoANextTracks->fGlobalTime[gamma2.currentSlot], // global time
            0.,                                                                 // local time
            secondaries.gammas.fSoANextTracks->fEventId[gamma2.currentSlot],
            secondaries.gammas.fSoANextTracks->fThreadId[gamma2.currentSlot], // eventID and threadID
            false,                                                            // whether this was the last step
            secondaries.gammas.fSoANextTracks->fStepCounter[gamma2.currentSlot]);
      }
    }

    // The current track is killed by not enqueuing into the next activeQueue.
    // slotManager.MarkSlotForFreeing(slot);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring,
                               soaTrack->fTrackId[slot],    // Track ID
                               soaTrack->fParentId[slot],   // parent Track ID
                               static_cast<short>(2),       // step limiting process ID
                               static_cast<char>(1),        // Particle type
                               elTrack.GetPStepLength(),    // Step length
                               energyDeposit,               // Total Edep
                               soaTrack->fWeight[slot],     // Track weight
                               soaTrack->fNavState[slot],   // Pre-step point navstate
                               currentTrack.preStepPos,     // Pre-step point position
                               currentTrack.preStepDir,     // Pre-step point momentum direction
                               currentTrack.preStepEKin,    // Pre-step point kinetic energy
                               currentTrack.nextState,      // Post-step point navstate
                               soaTrack->fPos[slot],        // Post-step point position
                               soaTrack->fDir[slot],        // Post-step point momentum direction
                               soaTrack->fEkin[slot],       // Post-step point kinetic energy
                               soaTrack->fGlobalTime[slot], // global time
                               soaTrack->fLocalTime[slot],  // local time
                               soaTrack->fEventId[slot], soaTrack->fThreadId[slot], // eventID and threadID
                               isLastStep,                                          // whether this was the last step
                               soaTrack->fStepCounter[slot]);                       // stepcounter
  }
}

template <typename Scoring>
__global__ void PositronStoppedAnnihilation(Track *electrons, SoATrack *soaTrack, G4HepEmElectronTrack *hepEMTracks,
                                            Secondaries secondaries, adept::MParray *nextActiveQueue,
                                            adept::MParray *interactingQueue, Scoring *userScoring,
                                            const bool returnAllSteps, const bool returnLastStep)
{
  int activeSize = interactingQueue->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    // const int slot           = (*active)[i];
    const int slot           = (*interactingQueue)[i];
    SlotManager &slotManager = *secondaries.positrons.fSlotManager;

    Track &currentTrack = electrons[slot];
    // the MCC vector is indexed by the logical volume id
    const int lvolID = soaTrack->fNavState[slot].GetLogicalId();

    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
    bool isLastStep           = true;

    // Retrieve HepEM track
    G4HepEmElectronTrack &elTrack = hepEMTracks[slot];
    G4HepEmTrack *theTrack        = elTrack.GetTrack();

    G4HepEmMSCTrackData *mscData = elTrack.GetMSCTrackData();
    G4HepEmRandomEngine rnge(&soaTrack->fRngState[slot]);

    double energyDeposit = theTrack->GetEnergyDeposit();

    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    // Annihilate the stopped positron into two gammas heading to opposite
    // directions (isotropic).

    PerformStoppedAnnihilation<false, Scoring>(slot, currentTrack, soaTrack, secondaries, energyDeposit, slotManager,
                                               ApplyCuts, theGammaCut, userScoring, returnLastStep);

    // Record the step. Edep includes the continuous energy loss and edep from secondaries which were cut
    if ((energyDeposit > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep)
      adept_scoring::RecordHit(userScoring,
                               soaTrack->fTrackId[slot],    // Track ID
                               soaTrack->fParentId[slot],   // parent Track ID
                               static_cast<short>(2),       // step limiting process ID
                               static_cast<char>(1),        // Particle type
                               elTrack.GetPStepLength(),    // Step length
                               energyDeposit,               // Total Edep
                               soaTrack->fWeight[slot],     // Track weight
                               soaTrack->fNavState[slot],   // Pre-step point navstate
                               currentTrack.preStepPos,     // Pre-step point position
                               currentTrack.preStepDir,     // Pre-step point momentum direction
                               currentTrack.preStepEKin,    // Pre-step point kinetic energy
                               currentTrack.nextState,      // Post-step point navstate
                               soaTrack->fPos[slot],        // Post-step point position
                               soaTrack->fDir[slot],        // Post-step point momentum direction
                               soaTrack->fEkin[slot],       // Post-step point kinetic energy
                               soaTrack->fGlobalTime[slot], // global time
                               soaTrack->fLocalTime[slot],  // local time
                               soaTrack->fEventId[slot], soaTrack->fThreadId[slot], // eventID and threadID
                               isLastStep,                                          // whether this was the last step
                               soaTrack->fStepCounter[slot]);                       // stepcounter
  }
}

} // namespace AsyncAdePT
