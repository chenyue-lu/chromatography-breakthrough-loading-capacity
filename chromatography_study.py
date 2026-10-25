"""Synthetic step loading of one hypothetical protein on one fixed column.

Run ``python chromatography_study.py --smoke`` in the project .venv.
No study results or figures are written. CADET's temporary HDF5 files are
created under tmp/simulation_files and removed by its Python interface.

Unit convention: m, s, mol, m^3; liquid concentrations are mol/m^3 liquid,
and q and q_max are mol/m^3 particle SOLID (excluding particle pores).
Flow enters the model through the mL/min -> m^3/s boundary in simulate_flow.
The CLI converts seconds to minutes for reporting only. Multiplying amounts
in mol by MOLAR_MASS_KG_MOL gives kg for later mass accounting.

Geometry and transport values are taken from the supplied CADET-Workshop
03.1.1_chromatography.ipynb, Example 2: Multi-component Langmuir Separation,
code cell 18 (zero-based). Attribution: CADET-Workshop contributors,
https://github.com/cadet/CADET-Workshop ; the upstream README specifies
CC BY 4.0: https://creativecommons.org/licenses/by/4.0/ . The local source
SHA256 is recorded below; this module does not require the notebook at runtime.
Changes from that example: one hypothetical protein, a constant step input,
kinetic binding with assumed rates/capacity/feed, and fully accessible pores.
Transferring its transport coefficients to this hypothetical system is a
modelling assumption, not evidence of measured protein/resin performance.
Fixed dispersion and film transfer ignore their possible dependence on flow.
The column is initially solute-free, isothermal, has spherical particles, and
uses CADET Danckwerts inlet and zero-gradient outlet boundary conditions.

Method references (separate from the workshop numerical parameter source):
https://cadet.github.io/v5.1.X/modelling/unit_operations/lumped_rate_model_with_pores.html
https://cadet.github.io/v5.1.X/modelling/binding/multi_component_langmuir.html
For this SINGLE component, dq/dt = k_a*c_p*(q_max - q) - k_d*q.
CADET's API calls the binding model MULTI_COMPONENT_LANGMUIR; NCOMP is one.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import random

SEED = 20261025
random.seed(SEED)

import numpy as np

np.random.seed(SEED)

PROJECT_DIR = Path(__file__).resolve().parent
# CADET-Process imports Matplotlib; keep its font/config cache local too.
os.environ["MPLCONFIGDIR"] = str(PROJECT_DIR / ".venv" / "matplotlib-cache")

import pandas as pd
import cadet_core
from CADETProcess import settings
from CADETProcess.processModel import (
    ComponentSystem,
    FlowSheet,
    Inlet,
    Langmuir,
    LumpedRateModelWithPores,
    Outlet,
    Process,
)
from CADETProcess.simulator import Cadet

FLOW_RATES_ML_MIN = (0.5, 1.0, 2.0)
OUTPUT_SAMPLING_LEVELS = (100, 200, 400)
BASELINE_OUTPUT_SAMPLING = 200
AXIAL_GRID_LEVELS = (20, 40, 80)
BASELINE_AXIAL_GRID = 40
SOLVER_TOLERANCE_LEVELS = ((1e-6, 1e-8), (1e-7, 1e-9), (1e-8, 1e-10))
BASELINE_SOLVER_TOLERANCES = (1e-7, 1e-9)  # (rtol, atol), native molar units

SECONDS_PER_MINUTE = 60.0
M3_PER_ML = 1e-6
COLUMN_LENGTH_M = 0.10
COLUMN_DIAMETER_M = 0.010
BED_VOLUME_M3 = np.pi * COLUMN_DIAMETER_M**2 * COLUMN_LENGTH_M / 4.0
BED_POROSITY = 0.37
PARTICLE_POROSITY = 0.33
PARTICLE_RADIUS_M = 4.5e-5
AXIAL_DISPERSION_M2_S = 2e-7
FILM_MASS_TRANSFER_M_S = 1e-4
PORE_ACCESSIBILITY = 1.0
MOLAR_MASS_KG_MOL = 50.0  # hypothetical 50 kDa protein
FEED_CONCENTRATION_MOL_M3 = 0.020  # equivalent to 1 kg/m^3 = 1 mg/mL
LANGMUIR_CAPACITY_MOL_M3_SOLID = 2.0  # equivalent to 100 kg/m^3 solid
ADSORPTION_RATE_M3_MOL_S = 0.050
DESORPTION_RATE_S = 1e-4
EQUILIBRIUM_COEFFICIENT_M3_MOL = ADSORPTION_RATE_M3_MOL_S / DESORPTION_RATE_S
HOLD_UP_VOLUME_M3 = BED_VOLUME_M3 * (
    BED_POROSITY + (1.0 - BED_POROSITY) * PARTICLE_POROSITY
)

# Duration is a stopping control, not a study variable. Each longer attempt
# restarts the same clean-column load at t=0, with an unchanged sampling step.
INITIAL_LOAD_RESIDENCE_TIMES = 8
MAX_LOAD_EXTENSIONS = 6  # fail explicitly if even 512 residence times are insufficient
TARGET_OUTLET_RATIO = 0.95
# Allow at most this dimensionless drawdown from ANY previous raw ratio.
# This validation threshold does not clip, smooth, or replace solver output.
RAW_MONOTONICITY_ATOL = 1e-6

SOURCE_NOTEBOOK_SHA256 = "bbabfa9503e24963ee8cbb306bf4887aa14031835d5ea84f0a9ede5d11572246"
WORKSHOP_SOURCE = (
    "CADET-Workshop contributors, 03.1.1_chromatography.ipynb, Example 2, "
    f"code cell 18 (zero-based), SHA256 {SOURCE_NOTEBOOK_SHA256}"
)

PARAMETER_RECORDS = pd.DataFrame.from_records(
    [
        ("Column length", "L", COLUMN_LENGTH_M, "m", "column.length / COL_LENGTH",
         "sourced", WORKSHOP_SOURCE + "; column.length = 0.1 m, unchanged."),
        ("Column diameter", "d", COLUMN_DIAMETER_M, "m", "column.diameter / CROSS_SECTION_AREA",
         "sourced", WORKSHOP_SOURCE + "; column.diameter = 0.01 m, unchanged; area = pi*d^2/4."),
        ("Packed-bed volume", "V_bed", BED_VOLUME_M3, "m^3 bed", "derived from column geometry",
         "derived", "Cylinder volume pi*d^2*L/4; includes interstitial and particle volumes."),
        ("Interstitial porosity", "epsilon_c", BED_POROSITY, "1", "column.bed_porosity / COL_POROSITY",
         "sourced", WORKSHOP_SOURCE + "; column.bed_porosity = 0.37, unchanged."),
        ("Particle porosity", "epsilon_p", PARTICLE_POROSITY, "1", "column.particle_porosity / PAR_POROSITY",
         "sourced", WORKSHOP_SOURCE + "; column.particle_porosity = 0.33, unchanged."),
        ("Particle radius", "r_p", PARTICLE_RADIUS_M, "m", "column.particle_radius / PAR_RADIUS",
         "sourced", WORKSHOP_SOURCE + "; column.particle_radius = 4.5e-5 m, unchanged."),
        ("Axial dispersion", "D_ax", AXIAL_DISPERSION_M2_S, "m^2/s", "column.axial_dispersion / COL_DISPERSION",
         "adapted", WORKSHOP_SOURCE + "; unchanged 2e-7 m^2/s transferred to hypothetical protein; fixed at all flows."),
        ("Film mass transfer", "k_f", FILM_MASS_TRANSFER_M_S, "m/s", "column.film_diffusion / FILM_DIFFUSION",
         "adapted", WORKSHOP_SOURCE + "; equal [1e-4, 1e-4] m/s entries reduced to one assumed protein; fixed at all flows."),
        ("Pore accessibility", "F_acc", PORE_ACCESSIBILITY, "1", "column.pore_accessibility / PORE_ACCESSIBILITY",
         "assumed", "All intraparticle pore liquid is accessible to the hypothetical protein."),
        ("Protein molar mass", "M", MOLAR_MASS_KG_MOL, "kg/mol", "reporting only; not a CADET input",
         "assumed", "Hypothetical 50 kDa protein; used only to convert molar amounts to mass."),
        ("Feed concentration", "C_F", FEED_CONCENTRATION_MOL_M3, "mol/m^3 liquid", "feed.c / CONST_COEFF",
         "assumed", "Fixed hypothetical molar step; C_F*M = 1 kg/m^3, equivalent to 1 mg/mL."),
        ("Langmuir capacity", "q_max", LANGMUIR_CAPACITY_MOL_M3_SOLID, "mol/m^3 solid", "binding.capacity / MCL_QMAX",
         "assumed", "Hypothetical solid-phase saturation loading, not DBC per packed-bed volume."),
        ("Adsorption rate", "k_a", ADSORPTION_RATE_M3_MOL_S, "m^3/(mol*s)", "binding.adsorption_rate / MCL_KA",
         "assumed", "Hypothetical finite-rate adsorption in dq/dt = k_a*c_p*(q_max-q)-k_d*q."),
        ("Desorption rate", "k_d", DESORPTION_RATE_S, "1/s", "binding.desorption_rate / MCL_KD",
         "assumed", "Hypothetical finite-rate desorption; no fitted or measured kinetics."),
        ("Equilibrium coefficient", "K_eq", EQUILIBRIUM_COEFFICIENT_M3_MOL, "m^3/mol", "binding.k_eq = k_a/k_d",
         "derived", "k_a/k_d; q_eq = q_max*K_eq*c_p/(1+K_eq*c_p)."),
        ("Initial bulk liquid", "c_0", 0.0, "mol/m^3 liquid", "column.c / INIT_C",
         "assumed", "Initially solute-free interstitial liquid at every axial cell."),
        ("Initial pore liquid", "c_p0", 0.0, "mol/m^3 liquid", "column.cp / INIT_CP",
         "assumed", "Initially solute-free pore liquid at every axial cell."),
        ("Initial adsorbed loading", "q_0", 0.0, "mol/m^3 solid", "column.q / INIT_Q",
         "assumed", "Initially unloaded solid phase at every axial cell."),
        ("Column hold-up volume", "V_H", HOLD_UP_VOLUME_M3, "m^3 liquid", "derived; column-only hold-up",
         "derived", "V_bed*[epsilon_c+(1-epsilon_c)*epsilon_p]; no extra-column volume."),
        *[("Flow rate", f"Q_{q:g}", q, "mL/min", "feed.flow_rate / CONNECTIONS (converted to m^3/s)",
           "sourced", "Approved brief, Mandatory MVP Scope item 2: exactly 0.5, 1.0, 2.0 mL/min.")
          for q in FLOW_RATES_ML_MIN],
    ],
    columns=("name", "symbol", "value", "unit", "cadet_mapping", "provenance", "source_or_rationale"),
)


@dataclass(frozen=True)
class SimulationResult:
    """Raw single-component states; spatial arrays have shape (time, axial cell).

    FV cells have equal geometric volumes. Later column mass in kg is
    M*V_cell*sum_z[epsilon_c*c_bulk + (1-epsilon_c)*epsilon_p*c_pore
                 + (1-epsilon_c)*(1-epsilon_p)*q].
    Concentrations and times are unmodified solver outputs. The retained
    inlet/outlet series also allow cumulative feed/effluent integration.
    """

    flow_ml_min: float
    flow_m3_s: float
    samples_per_residence_time: int
    axial_cells: int
    rtol: float
    atol: float
    load_extensions: int
    time_s: np.ndarray
    inlet_mol_m3: np.ndarray
    outlet_mol_m3: np.ndarray
    bulk_liquid_mol_m3: np.ndarray
    pore_liquid_mol_m3: np.ndarray
    adsorbed_mol_m3_solid: np.ndarray
    axial_positions_m: np.ndarray
    cell_bed_volume_m3: float

    @property
    def outlet_ratio(self) -> np.ndarray:
        """Raw outlet divided by the prescribed, constant feed concentration."""
        return self.outlet_mol_m3 / FEED_CONCENTRATION_MOL_M3


def build_process(flow_m3_s: float, axial_cells: int, duration_s: float) -> Process:
    """Build this study's fixed, clean-column constant step in native SI units."""
    components = ComponentSystem(["hypothetical_protein"])
    binding = Langmuir(components, name="langmuir")
    binding.is_kinetic = True
    binding.adsorption_rate = [ADSORPTION_RATE_M3_MOL_S]
    binding.desorption_rate = [DESORPTION_RATE_S]
    binding.capacity = [LANGMUIR_CAPACITY_MOL_M3_SOLID]

    feed = Inlet(components, name="feed")
    feed.c = [FEED_CONCENTRATION_MOL_M3]
    feed.flow_rate = flow_m3_s
    column = LumpedRateModelWithPores(components, name="column", discretization_scheme="FV")
    column.length = COLUMN_LENGTH_M
    column.diameter = COLUMN_DIAMETER_M
    column.bed_porosity = BED_POROSITY
    column.particle_porosity = PARTICLE_POROSITY
    column.particle_radius = PARTICLE_RADIUS_M
    column.axial_dispersion = AXIAL_DISPERSION_M2_S
    column.film_diffusion = [FILM_MASS_TRANSFER_M_S]
    column.pore_accessibility = [PORE_ACCESSIBILITY]
    column.binding_model = binding
    column.c = [0.0]
    column.cp = [0.0]
    column.q = [0.0]
    column.discretization.ncol = axial_cells
    column.discretization.par_geom = "SPHERE"
    column.solution_recorder.write_solution_bulk = True
    column.solution_recorder.write_solution_particle = True
    column.solution_recorder.write_solution_solid = True
    outlet = Outlet(components, name="outlet")

    flow_sheet = FlowSheet(components)
    flow_sheet.add_unit(feed, feed_inlet=True)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet, product_outlet=True)
    flow_sheet.add_connection(feed, column)
    flow_sheet.add_connection(column, outlet)
    process = Process(flow_sheet, name="hypothetical_step_load")
    process.cycle_time = duration_s  # one load interval; no cycling or feed events
    return process


def validate_raw_curve(result: SimulationResult, *, require_target: bool = True) -> float:
    """Reject invalid raw output; return the greatest drawdown in C_out/C_F.

    Nondecreasing means no raw ratio is more than RAW_MONOTONICITY_ATOL
    below any earlier ratio. No monotonicity correction is applied.
    """
    arrays = (result.time_s, result.inlet_mol_m3, result.outlet_mol_m3,
              result.bulk_liquid_mol_m3, result.pore_liquid_mol_m3,
              result.adsorbed_mol_m3_solid, result.axial_positions_m)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("Nonfinite raw time, concentration, or column state.")
    if result.time_s[0] != 0.0 or np.any(np.diff(result.time_s) <= 0.0):
        raise ValueError("Output time must start at zero and increase strictly.")
    if not np.allclose(result.inlet_mol_m3, FEED_CONCENTRATION_MOL_M3, rtol=0, atol=1e-12):
        raise ValueError("Inlet is not the prescribed constant-concentration step.")
    ratio = result.outlet_ratio
    drawdown = float(np.max(np.maximum.accumulate(ratio) - ratio))
    if drawdown > RAW_MONOTONICITY_ATOL:
        raise ValueError(f"Raw ratio drawdown {drawdown:.3g} exceeds {RAW_MONOTONICITY_ATOL:g}.")
    if np.min(ratio) < -RAW_MONOTONICITY_ATOL or np.max(ratio) > 1.0 + RAW_MONOTONICITY_ATOL:
        raise ValueError("Raw outlet ratio lies outside [0, 1] beyond numerical tolerance.")
    if require_target:
        if not np.any((ratio[:-1] < 0.10) & (ratio[1:] >= 0.10)):
            raise ValueError("Raw curve does not bracket the first 10% breakthrough.")
        if ratio[-1] < TARGET_OUTLET_RATIO:
            raise ValueError("Raw curve has not reached the required 0.95 outlet ratio.")
    return drawdown


def simulate_flow(
    flow_ml_min: float,
    *,
    samples_per_residence_time: int,
    axial_cells: int,
    rtol: float,
    atol: float,
) -> SimulationResult:
    """Run one prescribed flow with explicit numerical settings, extending as needed.

    Longer attempts rerun from the same zero initial state, doubling the load
    duration but retaining dt = (V_bed/Q)/samples_per_residence_time. Only the
    final complete trajectory is returned; attempts are never concatenated.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    if flow_ml_min not in FLOW_RATES_ML_MIN:
        raise ValueError(f"Flow must be one of {FLOW_RATES_ML_MIN} mL/min.")
    if samples_per_residence_time not in OUTPUT_SAMPLING_LEVELS:
        raise ValueError(f"Sampling must be one of {OUTPUT_SAMPLING_LEVELS}.")
    if axial_cells not in AXIAL_GRID_LEVELS:
        raise ValueError(f"Axial cells must be one of {AXIAL_GRID_LEVELS}.")
    if (rtol, atol) not in SOLVER_TOLERANCE_LEVELS:
        raise ValueError(f"(rtol, atol) must be one of {SOLVER_TOLERANCE_LEVELS}.")

    flow_m3_s = flow_ml_min * M3_PER_ML / SECONDS_PER_MINUTE
    residence_time_s = BED_VOLUME_M3 / flow_m3_s
    settings.temp_dir = str(PROJECT_DIR / "tmp")
    settings.save_log = False
    simulator = Cadet(
        install_path=Path(cadet_core.__file__).resolve().parent,
        temp_dir=PROJECT_DIR / "tmp" / "simulation_files",
    )
    simulator.use_dll = False  # execute the installed CADET-Core CLI
    simulator.solver_parameters.nthreads = 1
    simulator.time_resolution = residence_time_s / samples_per_residence_time
    simulator.time_integrator_parameters.reltol = rtol
    simulator.time_integrator_parameters.abstol = atol
    simulator.timeout = 120.0  # seconds per attempt; fail rather than hang

    for extension in range(MAX_LOAD_EXTENSIONS + 1):
        duration_s = INITIAL_LOAD_RESIDENCE_TIMES * 2**extension * residence_time_s
        process = build_process(flow_m3_s, axial_cells, duration_s)
        simulation = simulator.simulate(process)
        states = simulation.solution_cycles["column"]
        time = np.asarray(states["outlet"][0].time, dtype=float).copy()
        # One component, one particle type and one bound state: flatten only
        # singleton component/particle axes, preserving every time and FV cell.
        result = SimulationResult(
            flow_ml_min=flow_ml_min, flow_m3_s=flow_m3_s,
            samples_per_residence_time=samples_per_residence_time,
            axial_cells=axial_cells, rtol=rtol, atol=atol, load_extensions=extension,
            time_s=time,
            inlet_mol_m3=np.asarray(states["inlet"][0].solution).reshape(len(time)).copy(),
            outlet_mol_m3=np.asarray(states["outlet"][0].solution).reshape(len(time)).copy(),
            bulk_liquid_mol_m3=np.asarray(states["bulk"][0].solution).reshape(len(time), axial_cells).copy(),
            pore_liquid_mol_m3=np.asarray(states["particle"][0].solution).reshape(len(time), axial_cells).copy(),
            adsorbed_mol_m3_solid=np.asarray(states["solid"][0].solution).reshape(len(time), axial_cells).copy(),
            axial_positions_m=np.asarray(states["bulk"][0].axial_coordinates).copy(),
            cell_bed_volume_m3=BED_VOLUME_M3 / axial_cells,
        )
        validate_raw_curve(result, require_target=False)
        if result.outlet_ratio[-1] >= TARGET_OUTLET_RATIO:
            validate_raw_curve(result)
            return result
    raise RuntimeError(
        f"Q={flow_ml_min:g} mL/min did not reach {TARGET_OUTLET_RATIO} after "
        f"{MAX_LOAD_EXTENSIONS} duration extensions; final ratio={result.outlet_ratio[-1]:.6g}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="check all three baseline simulated loads; save no results")
    args = parser.parse_args()
    if not args.smoke:
        parser.print_help()
        return
    print(f"SIMULATED hypothetical protein; seed={SEED}; raw ratio tolerance={RAW_MONOTONICITY_ATOL:g}", flush=True)
    for flow in FLOW_RATES_ML_MIN:
        result = simulate_flow(
            flow, samples_per_residence_time=BASELINE_OUTPUT_SAMPLING,
            axial_cells=BASELINE_AXIAL_GRID,
            rtol=BASELINE_SOLVER_TOLERANCES[0], atol=BASELINE_SOLVER_TOLERANCES[1],
        )
        drawdown = validate_raw_curve(result)
        print(
            f"PASS Q={flow:g} mL/min: finite raw states; crosses 0.10; "
            f"final C_out/C_F={result.outlet_ratio[-1]:.6f}; "
            f"max drawdown={drawdown:.3g}; "
            f"end={result.time_s[-1] / SECONDS_PER_MINUTE:.3f} min; "
            f"samples={len(result.time_s)}; grid={result.axial_cells}; "
            f"duration extensions={result.load_extensions}", flush=True,
        )


if __name__ == "__main__":
    main()
