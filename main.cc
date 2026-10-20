#include <iostream>
#include "G4RunManager.hh"
#include "G4MTRunManager.hh"
#include "G4UImanager.hh"
#include "G4VisManager.hh"
#include "G4VisExecutive.hh"
#include "G4UIExecutive.hh"
#include "construction.hh"
#include "FTFP_BERT.hh"
#include "G4VModularPhysicsList.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4OpticalPhysics.hh"
#include "action.hh"
#include "G4SystemOfUnits.hh"
#include <ctime>
#include <string>

int main(int argc, char** argv) {
    G4long seed = time(NULL);
    G4Random::setTheSeed(seed);

#ifdef G4MULTITHREADED
    G4MTRunManager* runManager = new G4MTRunManager;
#else
    G4RunManager* runManager = new G4RunManager;
#endif

    // Setup initialization
    runManager->SetUserInitialization(new MyDetectorConstruction());
    G4VModularPhysicsList* physicsList = new FTFP_BERT();
    physicsList->RegisterPhysics(new G4StepLimiterPhysics());
    physicsList->RegisterPhysics( new G4OpticalPhysics ); //For optical photons
    runManager->SetUserInitialization(physicsList);
    runManager->SetUserInitialization(new MyActionInitialization());

    G4UImanager* UImanager = G4UImanager::GetUIpointer();

    G4String macroFile = "run/run.mac";
    bool visualization = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-vis") {
            visualization = true;
        } else {
            macroFile = arg;
        }
    }

    // Execute run command
    UImanager->ApplyCommand("/control/execute " + macroFile);

    // Visualization option
    if (visualization) {
        G4UIExecutive* ui = new G4UIExecutive(argc, argv);
        G4VisManager* visManager = new G4VisExecutive();
        visManager->Initialize();
        ui->SessionStart();
        delete visManager;
        delete ui;
    }

    delete runManager;
    return 0;
}
