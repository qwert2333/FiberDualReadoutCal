#ifndef ECALHIT_HH 
#define ECALHIT_HH 1

#include "G4VHit.hh"
#include "G4THitsCollection.hh"
#include "G4Allocator.hh"
#include "G4ThreeVector.hh"


class EcalHit : public G4VHit {
public:

    EcalHit() : fCellID(0), fLayer(0), fEdep(0.), fNph_scint(0), fNph_cherenkov(0) {}
    EcalHit(G4int cellID) : fCellID(cellID), fLayer(0), fEdep(0), fNph_scint(0), fNph_cherenkov(0) {}
    virtual ~EcalHit() = default;

    void setCellID(G4int _id) { fCellID = _id; }
    G4int GetCellID() const { return fCellID; }

    // Longitudinal (Z segmentation) index of this hit
    void setLayer(G4int _iz) { fLayer = _iz; }
    G4int GetLayer() const { return fLayer; }

    void setEdep(G4double _en) { fEdep = _en;}
    void addEdep(G4double _en) { fEdep += _en; }
    G4double GetEdep() const { return fEdep; }

    void setNphScint(G4int _nph) { fNph_scint = _nph;}
    void addNphScint(G4int _nph) { fNph_scint += _nph; }
    G4int GetNphScint() const { return fNph_scint; }    

    void setNphChren(G4int _nph) { fNph_cherenkov = _nph;}
    void addNphChren(G4int _nph) { fNph_cherenkov += _nph; }
    G4int GetNphChren() const { return fNph_cherenkov; }

private:
    G4int fCellID;     
    G4int fLayer;
    G4double fEdep;
    G4int fNph_scint;
    G4int fNph_cherenkov;

};

using EcalHitsCollection = G4THitsCollection<EcalHit>;


#endif
