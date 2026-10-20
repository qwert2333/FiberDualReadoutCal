#include "action.hh"

#include "MyEventAction.hh"
#include "MySteppingAction.hh"





MyActionInitialization::MyActionInitialization(){}

MyActionInitialization::~MyActionInitialization(){}

// called by runManager->SetUserInitialization(new MyActionInitialization()) in main.cc
void MyActionInitialization::BuildForMaster() const 
{
    MyRunAction *runAction = new MyRunAction();
    SetUserAction(runAction);
}

// called by runManager->SetUserInitialization(new MyActionInitialization()) in main.cc
void MyActionInitialization::Build() const 
{
    MyPrimaryGenerator *generator = new MyPrimaryGenerator();
    SetUserAction(generator);
    
    MyRunAction *runAction = new MyRunAction(generator);
    SetUserAction(runAction);

    MyEventAction *eventAction = new MyEventAction(runAction);
    SetUserAction(eventAction);

    MySteppingAction *steppingAction = new MySteppingAction( eventAction ) ;
    SetUserAction( steppingAction );


}
// --> instance made and methods called for both generator.cc and run.cc
