#include "analysis.hh"
#include "construction.hh"
#include "G4UnitsTable.hh"
#include "G4RunManager.hh"
#include "G4UImanager.hh"
#include "TString.h"
#include <cmath>
#include <cstdio>

MyRunAction::MyRunAction() : fPrimaryGen(nullptr)
{
    std::cout << " \n " << std::endl;
    std::cout << " -------  ENTER in MyRunAction::MyRunAction()  " << std::endl;
    std::cout << " \n " << std::endl;

    CreateEventNtuple();
}

MyRunAction::MyRunAction(MyPrimaryGenerator *PG): fPrimaryGen(PG)
{
    std::cout << " \n " << std::endl;
    std::cout << " -------  ENTER in MyRunAction::MyRunAction()  " << std::endl;
    std::cout << " -------  Read PrimaryGenerator for MC truth Info.  " << std::endl;
    std::cout << " \n " << std::endl;

    CreateEventNtuple();
}

void MyRunAction::CreateEventNtuple()
{
    G4AnalysisManager *man2 = G4AnalysisManager::Instance();

    // event-wise ntuple
    man2->CreateNtuple("eventTree", "eventTree");
    man2->CreateNtupleIColumn("eventID");
    man2->CreateNtupleSColumn("particle");
    man2->CreateNtupleDColumn("MCtruth_energy");
    man2->CreateNtupleDColumn("MCtruth_dir_x");
    man2->CreateNtupleDColumn("MCtruth_dir_y");
    man2->CreateNtupleDColumn("MCtruth_dir_z");
    man2->CreateNtupleDColumn("MCtruth_pos_x");
    man2->CreateNtupleDColumn("MCtruth_pos_y");
    man2->CreateNtupleDColumn("MCtruth_pos_z");

    man2->CreateNtupleDColumn("EdepScintFiber");
    man2->CreateNtupleDColumn("EdepCherenkovFiber");
    man2->CreateNtupleDColumn("EdepAbsorber");
    man2->CreateNtupleIColumn("Nph_Cherenkov");
    man2->CreateNtupleIColumn("Nph_Scintillation");
    man2->CreateNtupleIColumn("counter_Cerenkov_ScintFiber");
    man2->CreateNtupleIColumn("counter_Scintillation_ScintFiber");
    man2->CreateNtupleIColumn("counter_Cerenkov_CherenkovFiber");
    man2->CreateNtupleIColumn("counter_Scintillation_CherenkovFiber");
    man2->CreateNtupleDColumn("truthEdepScintTotal");
    man2->CreateNtupleDColumn("truthEdepScintEM");
    man2->CreateNtupleDColumn("truthEdepScintNonEM");
    man2->CreateNtupleDColumn("truthFemScint");
    man2->CreateNtupleDColumn("truthEdepCherenkovTotal");
    man2->CreateNtupleDColumn("truthEdepCherenkovEM");
    man2->CreateNtupleDColumn("truthEdepCherenkovNonEM");
    man2->CreateNtupleDColumn("truthFemCherenkov");
    man2->CreateNtupleDColumn("truthEdepAbsorberTotal");
    man2->CreateNtupleDColumn("truthEdepAbsorberEM");
    man2->CreateNtupleDColumn("truthEdepAbsorberNonEM");
    man2->CreateNtupleDColumn("truthFemAbsorber");
    man2->CreateNtupleDColumn("truthEdepTotal");
    man2->CreateNtupleDColumn("truthEdepEM");
    man2->CreateNtupleDColumn("truthEdepNonEM");
    man2->CreateNtupleDColumn("truthFem");

    // Geometry (same value in every event; lengths in mm)
    man2->CreateNtupleDColumn("geo_CaloSize");
    man2->CreateNtupleDColumn("geo_CaloLength");
    man2->CreateNtupleDColumn("geo_FiberDiameter");
    man2->CreateNtupleDColumn("geo_FiberPitch");
    man2->CreateNtupleIColumn("geo_NFiberPerSide");
    man2->CreateNtupleIColumn("geo_ZSegNum");
    man2->CreateNtupleDColumn("geo_ZSegLength");
    man2->CreateNtupleSColumn("geo_AbsorberMaterial");

    // Z segmentation: one entry per longitudinal segment, index = iz
    man2->CreateNtupleDColumn("Edep_ScintFiber_Layer", Edep_ScintFiber_Layer);
    man2->CreateNtupleDColumn("Edep_CherenkovFiber_Layer", Edep_CherenkovFiber_Layer);
    man2->CreateNtupleDColumn("Edep_Absorber_Layer", Edep_Absorber_Layer);
    man2->CreateNtupleIColumn("Nph_Cerenkov_ScintFiber_Layer", Nph_Cerenkov_ScintFiber_Layer);
    man2->CreateNtupleIColumn("Nph_Scintillation_ScintFiber_Layer", Nph_Scintillation_ScintFiber_Layer);
    man2->CreateNtupleIColumn("Nph_Cerenkov_CherenkovFiber_Layer", Nph_Cerenkov_CherenkovFiber_Layer);
    man2->CreateNtupleIColumn("Nph_Scintillation_CherenkovFiber_Layer", Nph_Scintillation_CherenkovFiber_Layer);

    // per fiber segment: cellID = (iz * NFiber + iy) * NFiber + ix, vecFiberType 0 = S, 1 = C
    man2->CreateNtupleIColumn("vecCellID", vecCellID );
    man2->CreateNtupleIColumn("vecFiberType", vecFiberType );
    man2->CreateNtupleDColumn("vecEdep", vecEdep );
    man2->CreateNtupleIColumn("vecNChren", vecNChren );
    man2->CreateNtupleIColumn("vecNScint", vecNScint );

    man2->FinishNtuple(0);
}

MyRunAction::~MyRunAction(){ fPrimaryGen = nullptr; }

void MyRunAction::BeginOfRunAction(const G4Run* run)
{

    std::cout << " \n " << std::endl;
    std::cout << " -------  ENTER in MyRunAction::BeginOfRunAction " << std::endl;
    std::cout << " \n " << std::endl;

    const MyDetectorConstruction* det = static_cast<const MyDetectorConstruction*>(
        G4RunManager::GetRunManager()->GetUserDetectorConstruction());

    if (!det) {
        G4Exception("MyRunAction", "NoDetector", FatalException, "DetectorConstruction not set!");
        return;
    }

    G4AnalysisManager *man3 = G4AnalysisManager::Instance();

    //Create the output Ntuple file
    char fiberTag[64];
    std::snprintf(fiberTag, sizeof(fiberTag), "_FiberD%.2fmm_Pitch%.2fmm",
                  det->GetFiberDiameter()/mm, det->GetFiberPitch()/mm);

    G4String OutFileName = "FiberDualReadoutCalo"; 
    OutFileName += "_Size" + std::to_string((int)std::lround(det->GetCaloSize()/mm)) + "x"
                + std::to_string((int)std::lround(det->GetCaloSize()/mm)) + "x"
                + std::to_string((int)std::lround(det->GetCaloLength()/mm)) + "mm";
    OutFileName += fiberTag;
    OutFileName += "_Abs" + det->GetAbsorberMaterialName();
    OutFileName += "_NZ" + std::to_string(det->GetZSegNum());
    if(fPrimaryGen){
      OutFileName += "_" + fPrimaryGen->GetParticleGun()->GetParticleDefinition()->GetParticleName();
      OutFileName += "_" + G4String(G4BestUnit(fPrimaryGen->GetParticleGun()->GetCurrentSource()->GetEneDist()->GetMonoEnergy(), "Energy"));
    }

    OutFileName += "_Run" + std::to_string(run->GetRunID()) + ".root";
    while (OutFileName.find(" ") != G4String::npos) {
      OutFileName.erase(OutFileName.find(" "), 1);
    }
    man3->OpenFile(OutFileName);
}

void MyRunAction::EndOfRunAction(const G4Run* run)
{

    std::cout << " \n " << std::endl;
    std::cout << " -------  ENTER in MyRunAction::EndOfRunAction " << std::endl;
    std::cout << " \n " << std::endl;

    G4AnalysisManager *man4 = G4AnalysisManager::Instance();
    man4->Write();
    man4->CloseFile();

   // Merge worker output files (same procedure as DualReadoutTileCal)
   if (G4RunManager::GetRunManager()->GetRunManagerType() != G4RunManager::workerRM)
   {
       G4cout << ">>> The run with RunID  " << run->GetRunID() << " has ended. <<<" << G4endl;
       G4UImanager *UImanager = G4UImanager::GetUIpointer();
       UImanager->ApplyCommand(Form("/control/shell ls *Run%i*.root | head -n 1 | sed 's/_Run[^.]*//' > .FileTemp", run->GetRunID()));
       
       UImanager->ApplyCommand("/control/shell tr -d '\\n' < .FileTemp > .FileTemp2");
       UImanager->ApplyCommand("/control/shell mv .FileTemp2 .FileTemp");
       
       UImanager->ApplyCommand(Form("/control/shell base=$(cat .FileTemp); hadd -f $base *Run%i_t*.root", run->GetRunID()));
       
       UImanager->ApplyCommand(Form("/control/shell rm *Run%i_t*.root", run->GetRunID()));
       
       UImanager->ApplyCommand("/control/shell rm .FileTemp");

   }

}

void MyRunAction::ResetEventData()
{
  Edep_ScintFiber_Layer.clear();
  Edep_CherenkovFiber_Layer.clear();
  Edep_Absorber_Layer.clear();
  Nph_Cerenkov_ScintFiber_Layer.clear();
  Nph_Scintillation_ScintFiber_Layer.clear();
  Nph_Cerenkov_CherenkovFiber_Layer.clear();
  Nph_Scintillation_CherenkovFiber_Layer.clear();

  vecCellID.clear();
  vecFiberType.clear();
  vecEdep.clear();
  vecNChren.clear();
  vecNScint.clear();
}
