#include "MySteppingAction.hh"

#include "MyEventAction.hh"
#include "construction.hh"

#include "G4Step.hh"
#include "globals.hh"

#include "G4Track.hh"
#include "G4ParticleDefinition.hh"
#include "G4VPhysicalVolume.hh"
#include "G4LogicalVolume.hh"
#include "G4SystemOfUnits.hh"

namespace {

enum class StepRegion { kScintFiber, kCherenkovFiber, kAbsorber, kOther };

StepRegion GetStepRegion(const G4Step* step)
{
        if (!step || !step->GetPreStepPoint()) return StepRegion::kOther;
        auto* volume = step->GetPreStepPoint()->GetPhysicalVolume();
        if (!volume || !volume->GetLogicalVolume()) return StepRegion::kOther;
        const G4String& name = volume->GetLogicalVolume()->GetName();
        if (name == MyDetectorConstruction::kScintFiberLogicalName) return StepRegion::kScintFiber;
        if (name == MyDetectorConstruction::kCherenkovFiberLogicalName) return StepRegion::kCherenkovFiber;
        // AbsorberLayerLogical, AbsorberRowEvenLogical, AbsorberRowOddLogical
        if (name.rfind("Absorber", 0) == 0) return StepRegion::kAbsorber;
        return StepRegion::kOther;
}

}



MySteppingAction::MySteppingAction(MyEventAction* eventAction) : fEventAction(eventAction) {}


void MySteppingAction::UserSteppingAction(const G4Step* step)
{
        G4Track *aTrack = step->GetTrack(); 
        const G4String particleName = aTrack->GetParticleDefinition()->GetParticleName();
        const StepRegion region = GetStepRegion(step);
        const G4double stepEnergyDeposit = step->GetTotalEnergyDeposit() /MeV;

        if (particleName != "opticalphoton") {
                fEventAction->IsEMComponentTrack(aTrack);
                if (region == StepRegion::kScintFiber) {
                        fEventAction->AddScintTruthEdep(aTrack, stepEnergyDeposit);
                }
                else if (region == StepRegion::kCherenkovFiber) {
                        fEventAction->AddCherenkovTruthEdep(aTrack, stepEnergyDeposit);
                }
                else if (region == StepRegion::kAbsorber) {
                        fEventAction->AddAbsorberTruthEdep(aTrack, stepEnergyDeposit);
                }
        }

        const    G4VProcess*  theprocess = aTrack->GetCreatorProcess();
        G4String CreatorprocessName = "None" ;
        if (theprocess != 0) {
                CreatorprocessName = theprocess->GetProcessName();
        }

        if ( particleName == "opticalphoton") {

                if ( CreatorprocessName == "Cerenkov") {
                        if (region == StepRegion::kScintFiber) {
                                fEventAction->increment_counter_Cerenkov_ScintFiber();
                        }
                        else if (region == StepRegion::kCherenkovFiber) {
                                fEventAction->increment_counter_Cerenkov_CherenkovFiber();
                        }
                }
                else {
                        if (region == StepRegion::kScintFiber) {
                                fEventAction->increment_counter_Scintillation_ScintFiber();
                        }
                        else if (region == StepRegion::kCherenkovFiber) {
                                fEventAction->increment_counter_Scintillation_CherenkovFiber();
                        }
                }

                // Do not track the optical photons: only the produced number is recorded.
                aTrack->SetTrackStatus(fStopAndKill);
        }

        // Register secondaries for the EM-component truth bookkeeping.
        // Optical photons never deposit energy here, so they are not registered.
        const std::vector< const G4Track * > *   secondaries = step -> GetSecondaryInCurrentStep () ;
        for (size_t isec=0; isec < secondaries->size(); isec++) {
                const G4Track* atrack = secondaries->at(isec);
                if (atrack->GetParticleDefinition()->GetParticleName() == "opticalphoton") continue;
                fEventAction->RegisterSecondaryTrack(aTrack, atrack);
        }
}
