# Fiber Dual-Readout Calorimeter Simulation

Geant4 simulation of a fiber-sampling dual-readout calorimeter, modelled on the
module described in S.Y. Jang et al., "Performance of dual-readout calorimeters
for various absorbers", NIM A 1090 (2026) 171598.  The framework (UI commands,
optical-photon counting, ntuple output, output naming, MT merging) follows
`../DualReadoutTileCal`.

Simplifications with respect to the paper: no fiber cladding, no photosensors,
no yellow filter, and no optical-photon transport.  Optical photons are killed
at their first step, so only the number of produced photons is recorded.

## Geometry

```text
World (air)
 └─ AbsorberLayer   absorber box  CaloSize x CaloSize x (CaloLength/ZSegNum)   copy = iz
     └─ AbsorberRow absorber box  (NFiber*pitch) x pitch x segment             copy = iy
         └─ ScintFiber / CherenkovFiber   G4Tubs, diameter FiberDiameter       copy = ix
```

- Fibers run along z over the full absorber length, on a square grid with
  centre-to-centre distance `FiberPitch`.
- `NFiber = floor(CaloSize / FiberPitch)` fibers per side; the grid is centred,
  the leftover margin is absorber.
- Scintillating and Cherenkov fibers alternate in a checkerboard:
  `(ix + iy) % 2 == 0` is scintillating.
- Calorimeter front face at z = 0; the beam goes along +z.
- Scintillating fiber: polystyrene card copied from DualReadoutTileCal
  (`Plastic`, Birks 0.126 mm/MeV).  Cherenkov fiber: PMMA, n = 1.49.

## Detector Configuration Commands

```text
/detector/CaloSize <value> mm          default 1000 mm (square cross section)
/detector/CaloLength <value> mm        default 2000 mm (fiber length is the same)
/detector/FiberDiameter <value> mm     default 1.0 mm  (paper, p. 2)
/detector/FiberPitch <value> mm        default 1.5 mm  (paper, p. 2)
/detector/ZSegNum <integer>            default 1 (no longitudinal segmentation)
/detector/AbsorberMaterial <name>      Cu (default), STAINLESS-STEEL, Fe, Pb, Tungsten, Brass
```

`FiberDiameter` must not exceed `FiberPitch`.

## Build and Run

```bash
source setup.sh            # or any environment with Geant4 + ROOT
mkdir -p build && cd build
cmake .. && make -j
./main run/run.mac         # add -vis for visualization
```

## Output ROOT Content

Tree `eventTree`, one row per event:

- primary truth: `particle`, `MCtruth_energy`, `MCtruth_dir_*`, `MCtruth_pos_*`;
- energy deposits: `EdepScintFiber`, `EdepCherenkovFiber`, `EdepAbsorber`;
- produced optical photons (from the hit collections, both fiber types summed):
  `Nph_Cherenkov`, `Nph_Scintillation`;
- produced optical photons per fiber type (stepping-action counters):
  `counter_Cerenkov_ScintFiber`, `counter_Scintillation_ScintFiber`,
  `counter_Cerenkov_CherenkovFiber`, `counter_Scintillation_CherenkovFiber`;
- truth EM / non-EM energy split, same definitions as DualReadoutTileCal:
  `truthEdepScint*`, `truthEdepCherenkov*`, `truthEdepAbsorber*`, `truthEdep*`, `truthFem*`;
- Z segmentation, vectors of length `ZSegNum` (index iz, iz = 0 at the front):
  `Edep_ScintFiber_Layer`, `Edep_CherenkovFiber_Layer`, `Edep_Absorber_Layer`,
  `Nph_Cerenkov_ScintFiber_Layer`, `Nph_Scintillation_ScintFiber_Layer`,
  `Nph_Cerenkov_CherenkovFiber_Layer`, `Nph_Scintillation_CherenkovFiber_Layer`;
- geometry (same in every event, lengths in mm): `geo_CaloSize`, `geo_CaloLength`,
  `geo_FiberDiameter`, `geo_FiberPitch`, `geo_NFiberPerSide`, `geo_ZSegNum`,
  `geo_ZSegLength`, `geo_AbsorberMaterial`;
- per fiber segment with a hit: `vecCellID` with
  `cellID = (iz * NFiber + iy) * NFiber + ix`, `vecFiberType` (0 = S, 1 = C),
  `vecEdep`, `vecNChren`, `vecNScint`.  Fiber position from the geometry columns:
  `x = (ix - (N-1)/2) * pitch`, `y = (iy - (N-1)/2) * pitch`,
  segment centre `z = (iz + 0.5) * geo_ZSegLength` (front face at z = 0).

Output file name example:

```text
FiberDualReadoutCalo_Size1000x1000x2000mm_FiberD1.00mm_Pitch1.50mm_AbsCu_NZ1_e-_10GeV.root
```
