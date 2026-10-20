#ifndef DETECTOR_HH
#define DETECTOR_HH

#include "G4VSensitiveDetector.hh"
#include "G4RunManager.hh"
#include "G4AnalysisManager.hh" 
#include "G4SystemOfUnits.hh"
#include "EcalHit.hh"
#include "G4SDManager.hh"

// One sensitive detector class for fibers and absorber.
//   kFiber   : one hit per (fiber, z segment), cellID = (iz * NFiber + iy) * NFiber + ix
//   kAbsorber: one hit per z segment,         cellID = iz
class MySensitiveDetector : public G4VSensitiveDetector
{
public:
    enum SDType { kFiber = 0, kAbsorber = 1 };

    MySensitiveDetector(G4String name, SDType type, G4int nFiberPerSide);
    ~MySensitiveDetector();
    void Initialize(G4HCofThisEvent *hitsCE) override;
    virtual G4bool ProcessHits(G4Step *, G4TouchableHistory *) override;
    
private:

    SDType fType;
    G4int fNFiberPerSide;
    EcalHitsCollection *fHitCollection; 
    G4int fHitCID; 
    G4int fHitIndex;
    std::map<G4int, G4int> cellIDCol; // <cellID, hit index>
    
};

#endif
