#ifndef RUN_HH
#define RUN_HH

#include "G4UserRunAction.hh"
#include "G4Run.hh"
#include "G4AnalysisManager.hh" 
#include "generator.hh"

#include <vector>

class MyRunAction : public G4UserRunAction
{
public:
    MyRunAction(MyPrimaryGenerator *PG);
    MyRunAction();
    ~MyRunAction();
    
    virtual void BeginOfRunAction(const G4Run*);
    virtual void EndOfRunAction(const G4Run*);
    void ResetEventData();
    MyPrimaryGenerator* GetGenerator() const {return fPrimaryGen;}

    // Per-layer (Z segmentation) vectors, index = iz. Assignment keeps the vector
    // objects bound to the ntuple columns.
    void Set_Edep_ScintFiber_Layer(const std::vector<G4double>& data) { Edep_ScintFiber_Layer = data; }
    void Set_Edep_CherenkovFiber_Layer(const std::vector<G4double>& data) { Edep_CherenkovFiber_Layer = data; }
    void Set_Edep_Absorber_Layer(const std::vector<G4double>& data) { Edep_Absorber_Layer = data; }
    void Set_Nph_Cerenkov_ScintFiber_Layer(const std::vector<G4int>& data) { Nph_Cerenkov_ScintFiber_Layer = data; }
    void Set_Nph_Scintillation_ScintFiber_Layer(const std::vector<G4int>& data) { Nph_Scintillation_ScintFiber_Layer = data; }
    void Set_Nph_Cerenkov_CherenkovFiber_Layer(const std::vector<G4int>& data) { Nph_Cerenkov_CherenkovFiber_Layer = data; }
    void Set_Nph_Scintillation_CherenkovFiber_Layer(const std::vector<G4int>& data) { Nph_Scintillation_CherenkovFiber_Layer = data; }

    // Per-fiber-segment vectors (one entry per hit fiber segment)
    void Fill_vecCellID(G4int data) { vecCellID.push_back(data); }
    void Fill_vecFiberType(G4int data) { vecFiberType.push_back(data); }
    void Fill_vecEdep(G4double data) { vecEdep.push_back(data); }
    void Fill_vecNChren(G4int data) { vecNChren.push_back(data); }
    void Fill_vecNScint(G4int data) { vecNScint.push_back(data); }

private: 
    void CreateEventNtuple();

    MyPrimaryGenerator *fPrimaryGen;

    std::vector<G4double> Edep_ScintFiber_Layer;
    std::vector<G4double> Edep_CherenkovFiber_Layer;
    std::vector<G4double> Edep_Absorber_Layer;
    std::vector<G4int> Nph_Cerenkov_ScintFiber_Layer;
    std::vector<G4int> Nph_Scintillation_ScintFiber_Layer;
    std::vector<G4int> Nph_Cerenkov_CherenkovFiber_Layer;
    std::vector<G4int> Nph_Scintillation_CherenkovFiber_Layer;

    std::vector<G4int> vecCellID;
    std::vector<G4int> vecFiberType;
    std::vector<G4double> vecEdep;
    std::vector<G4int> vecNChren;
    std::vector<G4int> vecNScint;

};

#endif
