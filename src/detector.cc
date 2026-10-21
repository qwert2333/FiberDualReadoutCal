#include "detector.hh"
#include "construction.hh"

#include <cmath>
#include <numeric>



MySensitiveDetector::MySensitiveDetector(G4String name, SDType type, G4int nFiberPerSide)
    : G4VSensitiveDetector(name),
      fType(type),
      fNFiberPerSide(nFiberPerSide),
      fHitCollection(nullptr),
      fHitCID(-1),
      fHitIndex(0) {
  collectionName.insert(name + "_hits");
}

MySensitiveDetector::~MySensitiveDetector(){ }

void MySensitiveDetector::Initialize(G4HCofThisEvent *hitsCE){
  fHitCollection = new EcalHitsCollection();
  if (fHitCID < 0)
  {
      fHitCID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  }
  hitsCE->AddHitsCollection(fHitCID, fHitCollection);
  fHitIndex = 0;
  cellIDCol.clear();
}


G4bool MySensitiveDetector::ProcessHits(G4Step *aStep, G4TouchableHistory *ROhist)
{

    G4double edep = aStep->GetTotalEnergyDeposit(); 
    G4Track *aTrack = aStep->GetTrack(); 
    G4String particleName = aTrack->GetParticleDefinition()->GetParticleName();

    // An optical photon absorbed on its first step (G4OpBoundaryProcess NoRINDEX at the
    // fiber/absorber boundary, or G4OpAbsorption) reports its own energy as a local
    // deposit. That energy is not subtracted from the parent's deposit, so it is not
    // added here; optical photons are only counted below (same as the truth Edep).
    if (particleName == "opticalphoton") edep = 0.;

    const    G4VProcess*  theprocess = aTrack->GetCreatorProcess();
    G4String CreatorprocessName = "None" ;
    if (theprocess != 0) 
      CreatorprocessName = theprocess->GetProcessName();

    // Volume hierarchy: World(depth 0) / AbsorberLayer(1) / AbsorberRow(2) / Fiber(3).
    // The layer (z segment) index is the copy number of the depth-1 volume.
    const G4VTouchable* touchable = aStep->GetPreStepPoint()->GetTouchable();
    const G4int historyDepth = touchable->GetHistoryDepth();
    const G4int iz = touchable->GetCopyNumber(historyDepth - 1);

    G4int cellID = iz;
    if (fType == kFiber) {
      const G4int ix = touchable->GetCopyNumber(0);
      const G4int iy = touchable->GetCopyNumber(1);
      cellID = (iz * fNFiberPerSide + iy) * fNFiberPerSide + ix;
    }

    EcalHit* hit = nullptr;
    if(cellIDCol.find(cellID) != cellIDCol.end()){
      hit = (*fHitCollection)[cellIDCol[cellID]];
      hit->addEdep(edep);
    }
    else{
      hit = new EcalHit(cellID);
      hit->setLayer(iz);
      hit->setEdep(edep);
      fHitCollection->insert(hit);
      cellIDCol.insert(std::make_pair(cellID, fHitIndex));
      fHitIndex++;
    }

    // Optical photons are killed at their first step in MySteppingAction,
    // so each produced photon is counted once, in the volume where it was created.
    if(particleName=="opticalphoton"){
      if(CreatorprocessName=="Cerenkov") {
        hit->addNphChren(1);
      }
      else {
        hit->addNphScint(1);
      }
    }
    return true;
}
