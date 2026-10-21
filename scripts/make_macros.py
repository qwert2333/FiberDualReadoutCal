#!/usr/bin/env python3
"""Generate Geant4 macros for the dual-readout study."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from common import config_dirs, energy_tag, iter_samples, load_config, particle_tag, repo_path, sample_name


def iter_macro_samples(config: dict) -> Iterable[dict]:
    """Yield fixed-energy samples and GPS energy-range samples."""
    for group in config["samples"]:
        has_energies = "energies_gev" in group
        has_range = "energy_range_gev" in group
        if has_energies == has_range:
            raise SystemExit(
                "Each sample must define exactly one of 'energies_gev' or "
                "'energy_range_gev'."
            )

        if has_energies:
            yield from iter_samples({"samples": [group]})
            continue

        energy_range = group["energy_range_gev"]
        if not isinstance(energy_range, dict):
            raise SystemExit("'energy_range_gev' must be an object with 'min' and 'max'.")
        try:
            energy_min = float(energy_range["min"])
            energy_max = float(energy_range["max"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit("'energy_range_gev.min' and '.max' must be numbers.") from exc
        if energy_min <= 0 or energy_max <= energy_min:
            raise SystemExit("GPS energy range must satisfy 0 < min < max.")

        energy_type = str(energy_range.get("type", "Lin"))
        if energy_type != "Lin":
            raise SystemExit("Only GPS energy range type 'Lin' is currently supported.")
        yield {
            "role": group["role"],
            "particle": group["particle"],
            "energy_min_gev": energy_min,
            "energy_max_gev": energy_max,
            "energy_type": energy_type,
            "events": int(group["events"]),
        }


def macro_sample_name(sample: dict) -> str:
    if "energy_gev" in sample:
        return sample_name(sample["role"], sample["particle"], sample["energy_gev"])
    range_tag = (
        f"{energy_tag(sample['energy_min_gev']).removesuffix('GeV')}to"
        f"{energy_tag(sample['energy_max_gev'])}"
    )
    return f"{sample['role']}_{particle_tag(sample['particle'])}_{range_tag}"


def gps_energy_commands(sample: dict) -> str:
    if "energy_gev" in sample:
        return f"/gps/ene/type Mono\n/gps/ene/mono {sample['energy_gev']:g} GeV"
    return (
        f"/gps/ene/type {sample['energy_type']}\n"
        f"/gps/ene/min {sample['energy_min_gev']:g} GeV\n"
        f"/gps/ene/max {sample['energy_max_gev']:g} GeV\n"
        "/gps/ene/gradient 0.\n"
        "/gps/ene/intercept 1."
    )


def macro_text(config, sample, sample_index: int) -> str:
    detector = config["detector"]
    seed1 = int(config.get("base_seed1", 2077)) + sample_index * 2
    seed2 = int(config.get("base_seed2", 1010)) + sample_index * 2
    energy_commands = gps_energy_commands(sample)
    return f"""/random/setSeeds {seed1} {seed2}

/run/numberOfThreads {int(config.get("threads", 1))}

/run/setCut {config.get("cut", "0.1 mm")}

/run/verbose {int(config.get("run_verbose", 1))}
/event/verbose {int(config.get("event_verbose", 0))}
/tracking/verbose {int(config.get("tracking_verbose", 0))}

# Detector construction (defaults match run/run.mac)
/detector/CaloSize {detector.get("calo_size", "1000 mm")}
/detector/CaloLength {detector.get("calo_length", "2000 mm")}
/detector/FiberDiameter {detector.get("fiber_diameter", "1.0 mm")}
/detector/FiberPitch {detector.get("fiber_pitch", "1.5 mm")}
/detector/ZSegNum {detector.get("zseg_num", "1")}
/detector/AbsorberMaterial {detector.get("absorber_mat", "Cu")}



/run/initialize

# Set particle gun
/gps/particle {sample["particle"]}
{energy_commands}
/gps/direction {config.get("gps_direction", "0 0 1")}
/gps/pos/type Plane
/gps/pos/shape Square
/gps/pos/centre {config.get("gps_position", "0 0 -50 cm")}
/gps/pos/halfx {config.get("gps_halfx", "1 cm")}
/gps/pos/halfy {config.get("gps_halfy", "1 cm")}

# Run event number
/run/beamOn {sample["events"]}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Study config, e.g. configs/dual_readout_smoke.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    dirs = config_dirs(config)

    rows = []
    for index, sample in enumerate(iter_macro_samples(config)):
        name = macro_sample_name(sample)
        path = dirs["macros"] / f"{name}.mac"
        path.write_text(macro_text(config, sample, index))
        rows.append((name, path))

    print(f"Generated {len(rows)} macros in {dirs['macros']}")
    for name, path in rows:
        print(f"  {name}: {path.relative_to(repo_path('.'))}")


if __name__ == "__main__":
    main()
