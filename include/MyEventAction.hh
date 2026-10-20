#ifndef MyEventAction_h
#define MyEventAction_h 1

#include "G4UserEventAction.hh"
#include "G4String.hh"
#include "analysis.hh"

#include <map>
#include <vector>

class G4Event;
class G4Track;


/// Event action class

class MyEventAction : public G4UserEventAction
{
  public:
    MyEventAction() : fRunAction(nullptr) { 
      ResetEventData(); 
      for (int i = 0; i < 3; ++i) fHitCollID[i] = -1;
    }
    MyEventAction(MyRunAction* runaction ) : fRunAction(runaction) { 
      ResetEventData(); 
      for (int i = 0; i < 3; ++i) fHitCollID[i] = -1;
    }
    ~MyEventAction() override = default;

    void BeginOfEventAction(const G4Event*) override;
    void EndOfEventAction(const G4Event*) override;

    int get_counter_Cerenkov_ScintFiber() const { return counter_Cerenkov_ScintFiber; }
    int get_counter_Scintillation_ScintFiber() const { return counter_Scintillation_ScintFiber; }
    int get_counter_Cerenkov_CherenkovFiber() const { return counter_Cerenkov_CherenkovFiber; }
    int get_counter_Scintillation_CherenkovFiber() const { return counter_Scintillation_CherenkovFiber; }
    void increment_counter_Cerenkov_ScintFiber() { counter_Cerenkov_ScintFiber++; }
    void increment_counter_Scintillation_ScintFiber() { counter_Scintillation_ScintFiber++; }
    void increment_counter_Cerenkov_CherenkovFiber() { counter_Cerenkov_CherenkovFiber++; }
    void increment_counter_Scintillation_CherenkovFiber() { counter_Scintillation_CherenkovFiber++; }

    G4bool IsEMComponentTrack(const G4Track* track);
    void RegisterSecondaryTrack(const G4Track* parent, const G4Track* secondary);
    void AddScintTruthEdep(const G4Track* track, G4double edep);
    void AddCherenkovTruthEdep(const G4Track* track, G4double edep);
    void AddAbsorberTruthEdep(const G4Track* track, G4double edep);
    void AddTotalTruthEdep(const G4Track* track, G4double edep);

    void ResetEventData(); 

    private :
    G4bool IsEMParticle(const G4String& particleName) const;
    G4bool IsNeutralMeson(const G4String& particleName) const;

    MyRunAction* fRunAction; 
    G4int fHitCollID[3];
    G4int eventID;
    G4String particle;
    G4double MCtruth_energy;
    G4double MCtruth_dir_x;
    G4double MCtruth_dir_y;
    G4double MCtruth_dir_z;
    G4double MCtruth_pos_x;
    G4double MCtruth_pos_y;
    G4double MCtruth_pos_z;
    G4double EdepScintFiber;
    G4double EdepCherenkovFiber;
    G4double EdepAbsorber;
    G4int Nph_Cherenkov;
    G4int Nph_Scint;

    G4double truthEdepScintTotal;
    G4double truthEdepScintEM;
    G4double truthEdepCherenkovTotal;
    G4double truthEdepCherenkovEM;
    G4double truthEdepAbsorberTotal;
    G4double truthEdepAbsorberEM;
    G4double truthEdepTotal;
    G4double truthEdepEM;
    std::map<G4int, G4bool> trackIsEM;
    std::map<G4int, G4String> trackParticleName;

    int counter_Cerenkov_ScintFiber;
    int counter_Scintillation_ScintFiber;
    int counter_Cerenkov_CherenkovFiber;
    int counter_Scintillation_CherenkovFiber;
};


#endif
