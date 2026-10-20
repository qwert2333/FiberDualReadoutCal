#ifndef MySteppingAction_h
#define MySteppingAction_h 1

#include "G4UserSteppingAction.hh"

class G4Step;
class MyEventAction;

/// Stepping action class.
///
/// Routes non-optical energy deposits to scintillating fibers, Cherenkov fibers
/// and absorber for the truth bookkeeping, counts produced optical photons per
/// fiber type, and kills optical photons after their first step (no transport).

class MySteppingAction : public G4UserSteppingAction
{
  public:
    MySteppingAction( MyEventAction* eventAction);
    ~MySteppingAction() override = default;

    void UserSteppingAction(const G4Step* step) override ;

  private:
    MyEventAction* fEventAction = nullptr;
};

#endif
