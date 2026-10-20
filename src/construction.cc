#include "construction.hh"
#include "G4Element.hh"
#include "G4Material.hh"
#include "G4SDManager.hh"

#include "G4GeometryManager.hh"
#include "G4PhysicalVolumeStore.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4SolidStore.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4Colour.hh"
#include "G4VisAttributes.hh"
#include "detector.hh"

#include "G4MaterialPropertiesTable.hh"

#include <algorithm>

const G4String MyDetectorConstruction::kScintFiberLogicalName = "ScintFiberLogical";
const G4String MyDetectorConstruction::kCherenkovFiberLogicalName = "CherenkovFiberLogical";

MyDetectorConstruction::MyDetectorConstruction() : G4VUserDetectorConstruction()
{
    nist = G4NistManager::Instance();
    logicAbsorberLayer = nullptr;
    logicAbsorberRowEven = nullptr;
    logicAbsorberRowOdd = nullptr;
    logicScintFiber = nullptr;
    logicCherenkovFiber = nullptr;
    matScintFiber = nullptr;
    matCherenkovFiber = nullptr;
    matAbsorber = nullptr;

    //Define a UI Messenger to allow runtime geometry / material selection
    fMessenger = new G4GenericMessenger(this, "/detector/", "Detector Control");

    DefineDim(fMessenger);
    DefineMaterials(fMessenger);

}



MyDetectorConstruction::~MyDetectorConstruction(){}



// =================================================================================
//
// ======                   Define dimensions
//
// =================================================================================


void MyDetectorConstruction::DefineDim(G4GenericMessenger* fMessenger)
{

    // Absorber block: 1 m x 1 m transverse, 2 m long (requested initial values).
    calo_size = 1000.*mm;
    calo_length = 2000.*mm;

    // Fiber diameter and centre-to-centre distance of neighbouring fibers,
    // from S.Y. Jang et al., NIM A 1090 (2026) 171598, Sec. 2.1 (p. 2) and Fig. 1(d) (p. 3):
    // "The diameter of these two types of fibers is 1 mm, and the distance between
    //  the centers of two neighboring fibers is 1.5 mm".
    fiber_diameter = 1.0*mm;
    fiber_pitch = 1.5*mm;

    // Longitudinal segmentation: 1 = no segmentation.
    nseg_z = 1;

    absorber_material_name = "Cu";

    fMessenger->DeclareMethodWithUnit("CaloSize", "mm", &MyDetectorConstruction::SetCaloSize)
        .SetGuidance("Set transverse size of the square absorber block")
        .SetParameterName("caloSize", true)
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethodWithUnit("CaloLength", "mm", &MyDetectorConstruction::SetCaloLength)
        .SetGuidance("Set longitudinal length of the absorber block (fiber length is the same)")
        .SetParameterName("caloLength", true)
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethodWithUnit("FiberDiameter", "mm", &MyDetectorConstruction::SetFiberDiameter)
        .SetGuidance("Set fiber diameter (scintillating and Cherenkov fibers)")
        .SetParameterName("fiberDiameter", true)
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethodWithUnit("FiberPitch", "mm", &MyDetectorConstruction::SetFiberPitch)
        .SetGuidance("Set centre-to-centre distance of neighbouring fibers")
        .SetParameterName("fiberPitch", true)
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethod("ZSegNum", &MyDetectorConstruction::SetZSeg)
        .SetGuidance("Set number of longitudinal segments")
        .SetParameterName("ZSegNum", true)
        .SetStates(G4State_PreInit, G4State_Idle);

    fMessenger->DeclareMethod("AbsorberMaterial", &MyDetectorConstruction::SetAbsorberMaterial)
        .SetGuidance("Set absorber material: Cu, STAINLESS-STEEL, Fe, Pb, Tungsten, or Brass")
        .SetParameterName("absorberMaterial", true)
        .SetCandidates("Cu STAINLESS-STEEL Fe Pb Tungsten Brass")
        .SetStates(G4State_PreInit, G4State_Idle);

}



// =================================================================================
//
// ======                   Define materials
//
// =================================================================================

// Scintillating fiber core: polystyrene. Same material card as DualReadoutTileCal
// (Plastic scintillator). Birks constant 0.126 mm/MeV is also the value quoted in
// the paper (Sec. 2.2, p. 3).
G4Material* MyDetectorConstruction::MakeScintillator(){
    auto* elH = nist->FindOrBuildElement("H");
    auto* elC = nist->FindOrBuildElement("C");

    const G4double density = 1.06 * g / cm3;
    auto* mat = new G4Material("G4_POLYSTYRENE", density, 2, kStateSolid);
    mat->AddElement(elH, 0.077418);
    mat->AddElement(elC, 0.922582);
    mat->GetIonisation()->SetMeanExcitationEnergy(68.7 * eV);
    mat->GetIonisation()->SetBirksConstant(0.126 * mm / MeV);

    const G4int nEntries = 3;
    G4double photonEnergy[nEntries] = {2.0 * eV, 2.5 * eV, 3.5 * eV};
    G4double rindex[nEntries] = {1.58, 1.58, 1.59};
    G4double absLength[nEntries] = {200. * cm, 200. * cm, 200. * cm};
    G4double scintSpectrum[nEntries] = {0.1, 1.0, 0.1};

    auto* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", photonEnergy, rindex, nEntries);
    mpt->AddProperty("ABSLENGTH", photonEnergy, absLength, nEntries);
    mpt->AddProperty("SCINTILLATIONCOMPONENT1", photonEnergy, scintSpectrum, nEntries);
    mpt->AddConstProperty("SCINTILLATIONYIELD", 62 / MeV);
    mpt->AddConstProperty("RESOLUTIONSCALE", 1.0);
    mpt->AddConstProperty("SCINTILLATIONTIMECONSTANT1", 2.1 * ns);
    mpt->AddConstProperty("SCINTILLATIONYIELD1", 1.0);
    mat->SetMaterialPropertiesTable(mpt);

    return mat;
}

// Cherenkov fiber core: PMMA, n = 1.49 (paper Sec. 2.2, p. 2).
// The photon-energy range 2.0-3.5 eV is taken to match the scintillator card above;
// it defines the wavelength window in which Cherenkov photons are generated.
G4Material* MyDetectorConstruction::MakePMMA(){
    auto* elH = nist->FindOrBuildElement("H");
    auto* elC = nist->FindOrBuildElement("C");
    auto* elO = nist->FindOrBuildElement("O");

    const G4double density = 1.19 * g / cm3;
    auto* mat = new G4Material("PMMA", density, 3, kStateSolid);
    mat->AddElement(elC, 5);
    mat->AddElement(elH, 8);
    mat->AddElement(elO, 2);

    const G4int nEntries = 3;
    G4double photonEnergy[nEntries] = {2.0 * eV, 2.5 * eV, 3.5 * eV};
    G4double rindex[nEntries] = {1.49, 1.49, 1.49};

    auto* mpt = new G4MaterialPropertiesTable();
    mpt->AddProperty("RINDEX", photonEnergy, rindex, nEntries);
    mat->SetMaterialPropertiesTable(mpt);

    return mat;
}

G4Material* MyDetectorConstruction::MakeBrass()
{
    auto* elCu = nist->FindOrBuildElement("Cu");
    auto* elZn = nist->FindOrBuildElement("Zn");

    const G4double density = 8.53 * g / cm3;
    auto* mat = new G4Material("Brass", density, 2, kStateSolid);
    mat->AddElement(elCu, 0.7);
    mat->AddElement(elZn, 0.3);

    return mat;
}

void MyDetectorConstruction::SetAbsorberMaterial(G4String materialName)
{
    if (materialName == absorber_material_name && matAbsorber != nullptr) {
        return;
    }

    if (materialName == "Cu") {
        matAbsorber = nist->FindOrBuildMaterial("G4_Cu");
    } else if (materialName == "STAINLESS-STEEL") {
        matAbsorber = nist->FindOrBuildMaterial("G4_STAINLESS-STEEL");
    } else if (materialName == "Fe") {
        matAbsorber = nist->FindOrBuildMaterial("G4_Fe");
    } else if (materialName == "Pb") {
        matAbsorber = nist->FindOrBuildMaterial("G4_Pb");
    } else if (materialName == "Tungsten") {
        matAbsorber = nist->FindOrBuildMaterial("G4_W");
    } else if (materialName == "Brass") {
        matAbsorber = MakeBrass();
    } else {
        G4Exception("MyDetectorConstruction::SetAbsorberMaterial",
                    "InvalidAbsorberMaterial", FatalException,
                    "Absorber material must be Cu, STAINLESS-STEEL, Fe, Pb, Tungsten, or Brass.");
    }

    absorber_material_name = materialName;

    for (auto* lv : {logicAbsorberLayer, logicAbsorberRowEven, logicAbsorberRowOdd}) {
        if (lv != nullptr) lv->SetMaterial(matAbsorber);
    }
}



void MyDetectorConstruction::DefineMaterials(G4GenericMessenger* fMessenger)
{
    auto air = nist->FindOrBuildMaterial("G4_AIR");

    // Define vacuum for the world volume
    matVacuum = nist->FindOrBuildMaterial("G4_Galactic");

    //matWorld = matVacuum;
    matWorld = air ;

    matScintFiber = MakeScintillator();
    matCherenkovFiber = MakePMMA();
    SetAbsorberMaterial(absorber_material_name);

}



// Construct detector geometry
G4VPhysicalVolume *MyDetectorConstruction::Construct()
{
    // Clean up previous geometries
    G4GeometryManager::GetInstance()->OpenGeometry();
    G4PhysicalVolumeStore::GetInstance()->Clean();
    G4LogicalVolumeStore::GetInstance()->Clean();
    G4SolidStore::GetInstance()->Clean();

    // ---- Sanity checks on the input parameters
    if (fiber_diameter <= 0. || fiber_pitch <= 0. || calo_size <= 0. || calo_length <= 0.) {
        G4Exception("MyDetectorConstruction::Construct", "InvalidGeometry", FatalException,
                    "CaloSize, CaloLength, FiberDiameter and FiberPitch must be positive.");
    }
    if (fiber_diameter > fiber_pitch) {
        G4Exception("MyDetectorConstruction::Construct", "FiberOverlap", FatalException,
                    "FiberDiameter must not exceed FiberPitch (fibers would overlap).");
    }
    if (nseg_z < 1) {
        G4Exception("MyDetectorConstruction::Construct", "InvalidZSeg", FatalException,
                    "ZSegNum must be >= 1.");
    }
    const G4int nfiber = GetFiberNumPerSide();
    if (nfiber < 1) {
        G4Exception("MyDetectorConstruction::Construct", "NoFiber", FatalException,
                    "CaloSize is smaller than FiberPitch: no fiber fits in the absorber.");
    }

    const G4double seg_length = calo_length / nseg_z;
    const G4double row_width = nfiber * fiber_pitch;

    // Define World, large enough for the calorimeter (front face at z = 0) and an upstream gun.
    worldSize = 2. * std::max(calo_size, calo_length) + 2. * m;
    solidWorld = new G4Box("solidWorld", worldSize/2, worldSize/2, worldSize/2);
    logicWorld = new G4LogicalVolume(solidWorld, matWorld, "logicWorld");
    physWorld  = new G4PVPlacement(0, G4ThreeVector(0., 0., 0.), logicWorld, "physWorld", 0, false, 0, true);
    logicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());

    // ---- Solids and logical volumes
    auto solidAbsorberLayer = new G4Box("AbsorberLayerSolid",
                                        calo_size/2., calo_size/2., seg_length/2.);
    auto solidAbsorberRow = new G4Box("AbsorberRowSolid",
                                      row_width/2., fiber_pitch/2., seg_length/2.);
    auto solidFiber = new G4Tubs("FiberSolid", 0., fiber_diameter/2., seg_length/2., 0., 360.*deg);

    logicAbsorberLayer = new G4LogicalVolume(solidAbsorberLayer, matAbsorber, "AbsorberLayerLogical");
    logicAbsorberRowEven = new G4LogicalVolume(solidAbsorberRow, matAbsorber, "AbsorberRowEvenLogical");
    logicAbsorberRowOdd = new G4LogicalVolume(solidAbsorberRow, matAbsorber, "AbsorberRowOddLogical");
    logicScintFiber = new G4LogicalVolume(solidFiber, matScintFiber, kScintFiberLogicalName);
    logicCherenkovFiber = new G4LogicalVolume(solidFiber, matCherenkovFiber, kCherenkovFiberLogicalName);

    logicAbsorberLayer->SetVisAttributes(new G4VisAttributes(G4Colour(0.72, 0.45, 0.2, 0.3)));
    logicAbsorberRowEven->SetVisAttributes(G4VisAttributes::GetInvisible());
    logicAbsorberRowOdd->SetVisAttributes(G4VisAttributes::GetInvisible());
    logicScintFiber->SetVisAttributes(new G4VisAttributes(G4Colour(1.0, 0.6, 0.0)));
    logicCherenkovFiber->SetVisAttributes(new G4VisAttributes(G4Colour(0.2, 0.3, 1.0)));

    // ---- Fibers inside the rows. Checkerboard pattern: (ix + iy) % 2 == 0 -> scintillating.
    //      Even rows (iy even) start with a scintillating fiber, odd rows with a Cherenkov fiber.
    const G4double x0 = -0.5 * row_width + 0.5 * fiber_pitch;
    for (G4int ix = 0; ix < nfiber; ++ix) {
      const G4double xPos = x0 + ix * fiber_pitch;
      const G4bool evenIx = (ix % 2 == 0);
      new G4PVPlacement(nullptr, G4ThreeVector(xPos, 0., 0.),
                        evenIx ? logicScintFiber : logicCherenkovFiber,
                        evenIx ? "ScintFiber" : "CherenkovFiber",
                        logicAbsorberRowEven, false, ix, false);
      new G4PVPlacement(nullptr, G4ThreeVector(xPos, 0., 0.),
                        evenIx ? logicCherenkovFiber : logicScintFiber,
                        evenIx ? "CherenkovFiber" : "ScintFiber",
                        logicAbsorberRowOdd, false, ix, false);
    }

    // ---- Rows inside one longitudinal layer
    const G4double y0 = -0.5 * row_width + 0.5 * fiber_pitch;
    for (G4int iy = 0; iy < nfiber; ++iy) {
      const G4double yPos = y0 + iy * fiber_pitch;
      const G4bool evenIy = (iy % 2 == 0);
      new G4PVPlacement(nullptr, G4ThreeVector(0., yPos, 0.),
                        evenIy ? logicAbsorberRowEven : logicAbsorberRowOdd,
                        evenIy ? "AbsorberRowEven" : "AbsorberRowOdd",
                        logicAbsorberLayer, false, iy, false);
    }

    // ---- Longitudinal layers (Z segmentation); calorimeter front face at z = 0
    for (G4int iz = 0; iz < nseg_z; ++iz) {
      const G4double zPos = (iz + 0.5) * seg_length;
      new G4PVPlacement(nullptr, G4ThreeVector(0., 0., zPos),
                        logicAbsorberLayer, "AbsorberLayer", logicWorld, false, iz, false);
    }

    // ---- Printout
    G4int nScint = 0, nCherenkov = 0;
    for (G4int iy = 0; iy < nfiber; ++iy) {
      for (G4int ix = 0; ix < nfiber; ++ix) {
        if ((ix + iy) % 2 == 0) nScint++;
        else nCherenkov++;
      }
    }
    const G4double fiberArea = CLHEP::pi * fiber_diameter * fiber_diameter / 4.;
    const G4double activeAreaFraction = (G4double)(nScint + nCherenkov) * fiberArea / (calo_size * calo_size);

    std::cout << "--- Create fiber dual-readout calorimeter: "
              << calo_size/mm << " mm * " << calo_size/mm << " mm * " << calo_length/mm << " mm"
              << ", front face at z = 0" << std::endl;
    std::cout << "---   Absorber material: " << absorber_material_name
              << ", X0 = " << matAbsorber->GetRadlen() / cm << " cm"
              << ", interaction length = " << matAbsorber->GetNuclearInterLength() / cm << " cm" << std::endl;
    std::cout << "---   Full depth = " << calo_length / matAbsorber->GetRadlen() << " X0(absorber), "
              << calo_length / matAbsorber->GetNuclearInterLength() << " lambda_I(absorber)" << std::endl;
    std::cout << "---   Fiber diameter = " << fiber_diameter/mm << " mm, pitch = " << fiber_pitch/mm
              << " mm, absorber wall between fibers = " << (fiber_pitch - fiber_diameter)/mm << " mm" << std::endl;
    std::cout << "---   Fiber grid: " << nfiber << " * " << nfiber << " = " << nfiber * nfiber
              << " fibers (" << nScint << " scintillating, " << nCherenkov << " Cherenkov)"
              << ", grid width = " << row_width/mm << " mm" << std::endl;
    std::cout << "---   Fiber cross-section area fraction (geometric) = " << activeAreaFraction << std::endl;
    std::cout << "---   Z segmentation: " << nseg_z << " segments of " << seg_length/mm << " mm" << std::endl;

    return physWorld;
}

// Construct sensitive detector
void MyDetectorConstruction::ConstructSDandField()
{
    const G4int nfiber = GetFiberNumPerSide();
    auto sdManager = G4SDManager::GetSDMpointer();

    auto sensDetScint = new MySensitiveDetector("ScintFiber", MySensitiveDetector::kFiber, nfiber);
    sdManager->AddNewDetector(sensDetScint);
    logicScintFiber->SetSensitiveDetector(sensDetScint);

    auto sensDetCherenkov = new MySensitiveDetector("CherenkovFiber", MySensitiveDetector::kFiber, nfiber);
    sdManager->AddNewDetector(sensDetCherenkov);
    logicCherenkovFiber->SetSensitiveDetector(sensDetCherenkov);

    auto sensDetAbsorber = new MySensitiveDetector("Absorber", MySensitiveDetector::kAbsorber, nfiber);
    sdManager->AddNewDetector(sensDetAbsorber);
    logicAbsorberLayer->SetSensitiveDetector(sensDetAbsorber);
    logicAbsorberRowEven->SetSensitiveDetector(sensDetAbsorber);
    logicAbsorberRowOdd->SetSensitiveDetector(sensDetAbsorber);

}
