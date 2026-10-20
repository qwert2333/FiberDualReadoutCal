#include "MyEventAction.hh"
#include "generator.hh"
#include "G4Event.hh"
#include "G4ios.hh"

#include "G4RunManager.hh"
#include "G4AnalysisManager.hh"
#include "G4Track.hh"

#include "construction.hh"
#include "G4SDManager.hh"



//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void MyEventAction::BeginOfEventAction(const G4Event* event) {

  ResetEventData();

  if ( fHitCollID[0]<0 )
  {
      auto sdManager = G4SDManager::GetSDMpointer();
      fHitCollID[0] = sdManager->GetCollectionID("ScintFiber_hits");
      fHitCollID[1] = sdManager->GetCollectionID("CherenkovFiber_hits");
      fHitCollID[2] = sdManager->GetCollectionID("Absorber_hits");
  }

  if(!fRunAction)
    fRunAction = static_cast<MyRunAction *>(const_cast<G4UserRunAction *>(G4RunManager::GetRunManager()->GetUserRunAction()));
  fRunAction->ResetEventData();

}

G4bool MyEventAction::IsEMParticle(const G4String& particleName) const
{
  return particleName == "gamma" || particleName == "e-" || particleName == "e+";
}

G4bool MyEventAction::IsNeutralMeson(const G4String& particleName) const
{
  return particleName == "pi0" || particleName == "eta" || particleName == "eta_prime";
}

G4bool MyEventAction::IsEMComponentTrack(const G4Track* track)
{
  if (!track) return false;

  const G4int trackID = track->GetTrackID();
  const G4String particleName = track->GetParticleDefinition()->GetParticleName();
  trackParticleName[trackID] = particleName;

  auto known = trackIsEM.find(trackID);
  if (known != trackIsEM.end()) {
    return known->second;
  }

  G4bool isEM = false;
  const G4int parentID = track->GetParentID();
  if (parentID == 0) {
    isEM = IsEMParticle(particleName);
  }
  else {
    auto parentEM = trackIsEM.find(parentID);
    if (parentEM != trackIsEM.end() && parentEM->second) {
      isEM = true;
    }
    else {
      auto parentName = trackParticleName.find(parentID);
      if (parentName != trackParticleName.end()) {
        isEM = IsNeutralMeson(parentName->second) && IsEMParticle(particleName);
      }
    }
  }

  trackIsEM[trackID] = isEM;
  return isEM;
}

void MyEventAction::RegisterSecondaryTrack(const G4Track* parent, const G4Track* secondary)
{
  if (!parent || !secondary) return;

  const G4int parentID = parent->GetTrackID();
  const G4int secondaryID = secondary->GetTrackID();
  const G4String parentName = parent->GetParticleDefinition()->GetParticleName();
  const G4String secondaryName = secondary->GetParticleDefinition()->GetParticleName();

  trackParticleName[parentID] = parentName;
  trackParticleName[secondaryID] = secondaryName;

  const G4bool parentEM = IsEMComponentTrack(parent);
  const G4bool secondaryEM = parentEM || (IsNeutralMeson(parentName) && IsEMParticle(secondaryName));
  trackIsEM[secondaryID] = secondaryEM;
}

void MyEventAction::AddTotalTruthEdep(const G4Track* track, G4double edep)
{
  if (edep <= 0.) return;
  truthEdepTotal += edep;
  if (IsEMComponentTrack(track)) {
    truthEdepEM += edep;
  }
}

void MyEventAction::AddScintTruthEdep(const G4Track* track, G4double edep)
{
  if (edep <= 0.) return;
  truthEdepScintTotal += edep;
  if (IsEMComponentTrack(track)) {
    truthEdepScintEM += edep;
  }
  AddTotalTruthEdep(track, edep);
}

void MyEventAction::AddCherenkovTruthEdep(const G4Track* track, G4double edep)
{
  if (edep <= 0.) return;
  truthEdepCherenkovTotal += edep;
  if (IsEMComponentTrack(track)) {
    truthEdepCherenkovEM += edep;
  }
  AddTotalTruthEdep(track, edep);
}

void MyEventAction::AddAbsorberTruthEdep(const G4Track* track, G4double edep)
{
  if (edep <= 0.) return;
  truthEdepAbsorberTotal += edep;
  if (IsEMComponentTrack(track)) {
    truthEdepAbsorberEM += edep;
  }
  AddTotalTruthEdep(track, edep);
}

//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......

void MyEventAction::EndOfEventAction(const G4Event* event)
{

  // periodic printing
  G4int eventID = event->GetEventID();
  if (eventID < 100 || eventID % 100 == 0) {
    G4cout << ">>> Event: " << eventID << G4endl;
  }

  const MyDetectorConstruction *detector = static_cast<const MyDetectorConstruction *>(G4RunManager::GetRunManager()->GetUserDetectorConstruction());
  const G4int nZ = detector->GetZSegNum();

  // Extract MC info
  MyPrimaryGenerator* fPrimaryGen = fRunAction->GetGenerator();
  if(fPrimaryGen){
    particle = fPrimaryGen->GetParticleGun()->GetParticleDefinition()->GetParticleName();
    MCtruth_energy = fPrimaryGen->GetParticleGun()->GetParticleEnergy();
    MCtruth_dir_x = fPrimaryGen->GetParticleGun()->GetParticleMomentumDirection().x();
    MCtruth_dir_y = fPrimaryGen->GetParticleGun()->GetParticleMomentumDirection().y();
    MCtruth_dir_z = fPrimaryGen->GetParticleGun()->GetParticleMomentumDirection().z();
    MCtruth_pos_x = fPrimaryGen->GetParticleGun()->GetParticlePosition().x();
    MCtruth_pos_y = fPrimaryGen->GetParticleGun()->GetParticlePosition().y();
    MCtruth_pos_z = fPrimaryGen->GetParticleGun()->GetParticlePosition().z();
  }

  // Per-layer (Z segmentation) sums, index = iz
  std::vector<G4double> edepScintLayer(nZ, 0.), edepCherenkovLayer(nZ, 0.), edepAbsorberLayer(nZ, 0.);
  std::vector<G4int> nphCerScintLayer(nZ, 0), nphSciScintLayer(nZ, 0);
  std::vector<G4int> nphCerCherenkovLayer(nZ, 0), nphSciCherenkovLayer(nZ, 0);

  //Get hit info
  G4HCofThisEvent *hitsCE = event->GetHCofThisEvent();
  EcalHitsCollection *ScintHitCollection = nullptr;
  EcalHitsCollection *CherenkovHitCollection = nullptr;
  EcalHitsCollection *AbsorberHitCollection = nullptr;
  if (hitsCE){
    ScintHitCollection = (EcalHitsCollection *)(hitsCE->GetHC(fHitCollID[0]));
    CherenkovHitCollection = (EcalHitsCollection *)(hitsCE->GetHC(fHitCollID[1]));
    AbsorberHitCollection = (EcalHitsCollection *)(hitsCE->GetHC(fHitCollID[2]));
  }

  if(ScintHitCollection){
    size_t n_hit = ScintHitCollection->entries();
    for(size_t i=0; i<n_hit; i++){
      const EcalHit* hit = (*ScintHitCollection)[i];
      EdepScintFiber += hit->GetEdep();
      Nph_Cherenkov += hit->GetNphChren();
      Nph_Scint += hit->GetNphScint();

      const G4int iz = hit->GetLayer();
      edepScintLayer[iz] += hit->GetEdep();
      nphCerScintLayer[iz] += hit->GetNphChren();
      nphSciScintLayer[iz] += hit->GetNphScint();

      fRunAction->Fill_vecCellID( hit->GetCellID() );
      fRunAction->Fill_vecFiberType( 0 );
      fRunAction->Fill_vecEdep(   hit->GetEdep() );
      fRunAction->Fill_vecNChren( hit->GetNphChren() );
      fRunAction->Fill_vecNScint( hit->GetNphScint() );
    }
  }

  if(CherenkovHitCollection){
    size_t n_hit = CherenkovHitCollection->entries();
    for(size_t i=0; i<n_hit; i++){
      const EcalHit* hit = (*CherenkovHitCollection)[i];
      EdepCherenkovFiber += hit->GetEdep();
      Nph_Cherenkov += hit->GetNphChren();
      Nph_Scint += hit->GetNphScint();

      const G4int iz = hit->GetLayer();
      edepCherenkovLayer[iz] += hit->GetEdep();
      nphCerCherenkovLayer[iz] += hit->GetNphChren();
      nphSciCherenkovLayer[iz] += hit->GetNphScint();

      fRunAction->Fill_vecCellID( hit->GetCellID() );
      fRunAction->Fill_vecFiberType( 1 );
      fRunAction->Fill_vecEdep(   hit->GetEdep() );
      fRunAction->Fill_vecNChren( hit->GetNphChren() );
      fRunAction->Fill_vecNScint( hit->GetNphScint() );
    }
  }

  if(AbsorberHitCollection){
    size_t n_hit = AbsorberHitCollection->entries();
    for(size_t i=0; i<n_hit; i++){
      const EcalHit* hit = (*AbsorberHitCollection)[i];
      EdepAbsorber += hit->GetEdep();
      edepAbsorberLayer[hit->GetLayer()] += hit->GetEdep();
    }
  }

  fRunAction->Set_Edep_ScintFiber_Layer(edepScintLayer);
  fRunAction->Set_Edep_CherenkovFiber_Layer(edepCherenkovLayer);
  fRunAction->Set_Edep_Absorber_Layer(edepAbsorberLayer);
  fRunAction->Set_Nph_Cerenkov_ScintFiber_Layer(nphCerScintLayer);
  fRunAction->Set_Nph_Scintillation_ScintFiber_Layer(nphSciScintLayer);
  fRunAction->Set_Nph_Cerenkov_CherenkovFiber_Layer(nphCerCherenkovLayer);
  fRunAction->Set_Nph_Scintillation_CherenkovFiber_Layer(nphSciCherenkovLayer);

  // fill the ntuple 
  G4AnalysisManager *man5 = G4AnalysisManager::Instance();
  G4int ntupleID = 0;
  int idx = 0;

  man5->FillNtupleIColumn( ntupleID, idx, eventID);  idx ++;
  man5->FillNtupleSColumn( ntupleID, idx, particle);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_energy);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_dir_x);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_dir_y);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_dir_z);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_pos_x);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_pos_y);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, MCtruth_pos_z);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, EdepScintFiber);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, EdepCherenkovFiber);  idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, EdepAbsorber);  idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, Nph_Cherenkov); idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, Nph_Scint);  idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, counter_Cerenkov_ScintFiber); idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, counter_Scintillation_ScintFiber); idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, counter_Cerenkov_CherenkovFiber); idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, counter_Scintillation_CherenkovFiber); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepScintTotal); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepScintEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepScintTotal - truthEdepScintEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepScintTotal > 0. ? truthEdepScintEM / truthEdepScintTotal : -1.); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepCherenkovTotal); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepCherenkovEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepCherenkovTotal - truthEdepCherenkovEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepCherenkovTotal > 0. ? truthEdepCherenkovEM / truthEdepCherenkovTotal : -1.); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepAbsorberTotal); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepAbsorberEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepAbsorberTotal - truthEdepAbsorberEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepAbsorberTotal > 0. ? truthEdepAbsorberEM / truthEdepAbsorberTotal : -1.); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepTotal); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepTotal - truthEdepEM); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, truthEdepTotal > 0. ? truthEdepEM / truthEdepTotal : -1.); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, detector->GetCaloSize()/mm); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, detector->GetCaloLength()/mm); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, detector->GetFiberDiameter()/mm); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, detector->GetFiberPitch()/mm); idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, detector->GetFiberNumPerSide()); idx ++;
  man5->FillNtupleIColumn( ntupleID, idx, detector->GetZSegNum()); idx ++;
  man5->FillNtupleDColumn( ntupleID, idx, detector->GetZSegLength()/mm); idx ++;
  man5->FillNtupleSColumn( ntupleID, idx, detector->GetAbsorberMaterialName()); idx ++;
  // vector columns are bound by reference in MyRunAction and filled above

  man5->AddNtupleRow( ntupleID);
}

void MyEventAction::ResetEventData()
{
  eventID = 0;
  particle = "";
  MCtruth_energy = 0;
  MCtruth_dir_x = 0;
  MCtruth_dir_y = 0;
  MCtruth_dir_z = 0;
  MCtruth_pos_x = 0;
  MCtruth_pos_y = 0;
  MCtruth_pos_z = 0;
  EdepScintFiber = 0.;
  EdepCherenkovFiber = 0.;
  EdepAbsorber = 0.;
  Nph_Cherenkov = 0;
  Nph_Scint = 0;

  truthEdepScintTotal = 0.;
  truthEdepScintEM = 0.;
  truthEdepCherenkovTotal = 0.;
  truthEdepCherenkovEM = 0.;
  truthEdepAbsorberTotal = 0.;
  truthEdepAbsorberEM = 0.;
  truthEdepTotal = 0.;
  truthEdepEM = 0.;
  trackIsEM.clear();
  trackParticleName.clear();

  counter_Cerenkov_ScintFiber = 0;
  counter_Scintillation_ScintFiber = 0;
  counter_Cerenkov_CherenkovFiber = 0;
  counter_Scintillation_CherenkovFiber = 0;
}
//....oooOO0OOooo........oooOO0OOooo........oooOO0OOooo........oooOO0OOooo......
