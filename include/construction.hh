#ifndef CONSTRUCTION_HH
#define CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4SystemOfUnits.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4GenericMessenger.hh"
#include "detector.hh"
#include <string>
#include <cmath>
#include "G4ThreeVector.hh"


// Fiber dual-readout calorimeter:
//   a square absorber block (default Cu, 1 m x 1 m x 2 m) with cylindrical
//   scintillating and Cherenkov fibers running along z, placed on a square grid
//   and alternating in a checkerboard pattern ((ix+iy)%2 == 0 -> scintillating).
//
// Volume hierarchy (keeps the number of placements small for ~4e5 fibers):
//   World
//    └─ AbsorberLayer  (absorber box, CaloSize x CaloSize x CaloLength/ZSegNum), copy = iz
//        └─ AbsorberRowEven / AbsorberRowOdd (absorber box, NFiber*pitch x pitch x seg), copy = iy
//            └─ ScintFiber / CherenkovFiber (G4Tubs, length = seg), copy = ix
//
// The calorimeter front face is at z = 0, the beam goes along +z.

class MyDetectorConstruction : public G4VUserDetectorConstruction
{
public:
    MyDetectorConstruction();
    ~MyDetectorConstruction();

    virtual G4VPhysicalVolume *Construct();

    G4Material* MakeScintillator();
    G4Material* MakePMMA();
    G4Material* MakeBrass();

    void DefineMaterials(G4GenericMessenger* fMessenger);
    void DefineDim(G4GenericMessenger* fMessenger);

    void SetCaloSize( G4double size ) { calo_size = size; }
    void SetCaloLength( G4double length ) { calo_length = length; }
    void SetFiberDiameter( G4double d ) { fiber_diameter = d; }
    void SetFiberPitch( G4double p ) { fiber_pitch = p; }
    void SetZSeg( G4int num ) { nseg_z = num; }
    void SetAbsorberMaterial( G4String materialName );

    G4double GetCaloSize() const { return calo_size; }
    G4double GetCaloLength() const { return calo_length; }
    G4double GetFiberDiameter() const { return fiber_diameter; }
    G4double GetFiberPitch() const { return fiber_pitch; }
    G4int GetZSegNum() const { return nseg_z; }
    G4double GetZSegLength() const { return calo_length / nseg_z; }
    // Number of fibers per side of the square grid (same in x and y).
    G4int GetFiberNumPerSide() const { return (G4int)std::floor(calo_size / fiber_pitch + 1e-9); }
    G4String GetAbsorberMaterialName() const { return absorber_material_name; }

    // Logical-volume names, used by the stepping action to route energy deposits.
    static const G4String kScintFiberLogicalName;
    static const G4String kCherenkovFiberLogicalName;

private:
    G4GenericMessenger *fMessenger;  // Messenger for UI commands
    G4NistManager* nist;

    //Dimensions
    G4double worldSize;

    G4double calo_size, calo_length;
    G4double fiber_diameter, fiber_pitch;
    G4int nseg_z;

    G4String absorber_material_name;

    G4Box       *solidWorld;

    G4LogicalVolume   *logicWorld;
    G4LogicalVolume   *logicAbsorberLayer;
    G4LogicalVolume   *logicAbsorberRowEven;
    G4LogicalVolume   *logicAbsorberRowOdd;
    G4LogicalVolume   *logicScintFiber;
    G4LogicalVolume   *logicCherenkovFiber;

    G4VPhysicalVolume  *physWorld;

    G4Material *matWorld ;
    G4Material *matVacuum;

    G4Material *matScintFiber;
    G4Material *matCherenkovFiber;
    G4Material *matAbsorber;

    virtual void ConstructSDandField();

};

#endif
