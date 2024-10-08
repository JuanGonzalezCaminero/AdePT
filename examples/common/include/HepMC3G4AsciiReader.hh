// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

//
/// \file eventgenerator/HepMC3/HepMCEx01/include/HepMC3G4AsciiReader.hh
/// \brief Definition of the HepMC3G4AsciiReader class
//
//

#ifndef HEPMC3_G4_ASCII_READER_H
#define HEPMC3_G4_ASCII_READER_H

#include "HepMC3G4Interface.hh"
#include "HepMC3/Print.h"

#include <climits>
#include <memory>

class HepMC3G4AsciiReaderMessenger;

class HepMC3G4AsciiReader : public HepMC3G4Interface {
protected:
  G4String fFilename;
  int fFirstEventNumber  = 0;
  int fMaxNumberOfEvents = INT_MAX;
  std::vector<HepMC3::GenEvent> fEvents;

  G4int fVerbose = 0;
  std::unique_ptr<HepMC3G4AsciiReaderMessenger> fMessenger;

  virtual HepMC3::GenEvent *GenerateHepMCEvent(int eventId);

public:
  HepMC3G4AsciiReader();
  ~HepMC3G4AsciiReader();

  // set/get methods
  void SetFileName(G4String name);
  G4String GetFileName() const;

  void SetVerboseLevel(G4int i);
  G4int GetVerboseLevel() const;

  void SetMaxNumberOfEvents(G4int i);
  G4int GetMaxNumberOfEvents() const;

  void SetFirstEventNumber(G4int i);
  G4int GetFirstEventNumber() const;

  void Initialize()
  {
    fEvents.clear();
    GenerateHepMCEvent(0);
  }

private:
  void Read();
};

// ====================================================================
// inline functions
// ====================================================================

inline void HepMC3G4AsciiReader::SetFileName(G4String name)
{
  fFilename = name;
}

inline G4String HepMC3G4AsciiReader::GetFileName() const
{
  return fFilename;
}

inline void HepMC3G4AsciiReader::SetVerboseLevel(G4int i)
{
  fVerbose = i;
}

inline G4int HepMC3G4AsciiReader::GetVerboseLevel() const
{
  return fVerbose;
}

inline void HepMC3G4AsciiReader::SetMaxNumberOfEvents(G4int i)
{
  fMaxNumberOfEvents = i;
}

inline G4int HepMC3G4AsciiReader::GetMaxNumberOfEvents() const
{
  return fMaxNumberOfEvents;
}

inline void HepMC3G4AsciiReader::SetFirstEventNumber(G4int i)
{
  fFirstEventNumber = i;
}

inline G4int HepMC3G4AsciiReader::GetFirstEventNumber() const
{
  return fFirstEventNumber;
}

#endif
