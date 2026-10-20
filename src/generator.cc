#include "generator.hh"
#include "G4RandomDirection.hh"
#include <math.h>

MyPrimaryGenerator::MyPrimaryGenerator() : G4VUserPrimaryGeneratorAction(), fParticleGun(new G4GeneralParticleSource())
{

    fParticleGun->SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("gamma"));
    fParticleGun->SetParticlePosition(G4ThreeVector(0,0,-50.*cm));

    //G4int numberOfShotFromParticleGun = 1; // Conventional to assume one primary paricle per one event.

    //fParticleGun    = new G4ParticleGun(numberOfShotFromParticleGun);

    //G4ParticleTable *particleTable = G4ParticleTable::GetParticleTable();
    //G4String particleName="anti_proton"; //"proton","neutron","gamma","e-","e+"
    //G4ParticleDefinition *particle = particleTable->FindParticle(particleName);
    //fParticleGun->SetParticleDefinition(particle);
}

MyPrimaryGenerator::~MyPrimaryGenerator()
{
    delete fParticleGun;
}

// Called by SetUserAction(generator) in action.cc
void MyPrimaryGenerator::GeneratePrimaries(G4Event *anEvent)
{
    fParticleGun->GeneratePrimaryVertex(anEvent);

    //G4ThreeVector pos = G4ThreeVector(0, 0, xposition);
    //fParticleGun->SetParticlePosition(pos);

    //const int numberOfPrimaries = 1;
    //for (int i = 0; i < numberOfPrimaries; i++){
    //    G4double xdir = G4RandGauss::shoot(0., 0.015);
    //    G4double ydir = G4RandGauss::shoot(0., 0.015);
    //    G4ThreeVector dir = G4ThreeVector(xdir, ydir, 1.); 
    //    fParticleGun->SetParticleMomentumDirection(dir);    // for G4Gun
    //    fParticleGun->GeneratePrimaryVertex(anEvent);
    //}
}




