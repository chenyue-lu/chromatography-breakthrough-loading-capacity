"""Synthetic step loading of one hypothetical protein on one fixed column.

Run ``python chromatography_study.py --run`` for the complete study, or
``python chromatography_study.py --smoke`` for baseline checks without outputs.
The complete study writes only its five specified outputs after acceptance.
CADET's temporary HDF5 files are
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
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import io
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
from scipy.integrate import cumulative_trapezoid
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
# Dimensionless floor; raw_ratio_tolerance also accounts for native solver atol.
# Validation never clips, smooths, or replaces solver output.
RAW_MONOTONICITY_ATOL = 1e-6
RAW_RATIO_ATOL_MULTIPLIER = 4.0
ACCEPTANCE_LIMIT = 0.01  # fraction, for both mass conservation and refinement
MG_PER_KG = 1e6
REFINEMENT_FACTORS = ("sampling", "axial_grid", "solver")
RESULT_FILENAMES = (
    "parameters.csv", "summary.csv", "breakthrough_curves.csv",
    "simulated_breakthrough_comparison.png", "simulated_t10_dbc10_comparison.png",
)

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


def raw_ratio_tolerance(atol_mol_m3: float) -> float:
    """Dimensionless max(1e-6, 4*atol/C_F), independent of observed output samples.

    A drawdown compares two independently approximate concentrations. Four
    absolute-tolerance units is a declared numerical screening allowance,
    not a rigorous error bound. It is 2e-6 at the loosest prescribed atol,
    and 1e-6 at the middle/finest levels; all are far below 1% breakthrough.
    """
    return max(RAW_MONOTONICITY_ATOL, RAW_RATIO_ATOL_MULTIPLIER * atol_mol_m3 / FEED_CONCENTRATION_MOL_M3)


def validate_raw_curve(result: SimulationResult, *, require_target: bool = True) -> float:
    """Reject invalid raw output; return the greatest drawdown in C_out/C_F.

    Nondecreasing means no raw ratio is more than raw_ratio_tolerance(atol)
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
    tolerance = raw_ratio_tolerance(result.atol)
    if drawdown > tolerance:
        raise ValueError(f"Raw ratio drawdown {drawdown:.3g} exceeds {tolerance:g}.")
    if np.min(ratio) < -tolerance or np.max(ratio) > 1.0 + tolerance:
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


@dataclass(frozen=True)
class Crossing:
    time_s: float
    left: int
    right: int
    fraction: float


@dataclass(frozen=True)
class NumericalSettings:
    samples_per_residence_time: int = BASELINE_OUTPUT_SAMPLING
    axial_cells: int = BASELINE_AXIAL_GRID
    rtol: float = BASELINE_SOLVER_TOLERANCES[0]
    atol: float = BASELINE_SOLVER_TOLERANCES[1]


@dataclass(frozen=True)
class RefinementResult:
    factor: str
    settings: tuple[NumericalSettings, ...]
    metrics: np.ndarray  # (coarse/middle/fine, flow rate, t10_s/DBC_mg_ml_bed)
    selected_index: int
    converged: bool
    disclosure: str


def first_crossing(time_s: np.ndarray, ratio: np.ndarray, target: float = 0.10) -> Crossing:
    """Linearly interpolate the FIRST upward bracket, including an exact right endpoint."""
    time_s, ratio = np.asarray(time_s, dtype=float), np.asarray(ratio, dtype=float)
    if (time_s.ndim != 1 or ratio.shape != time_s.shape or len(time_s) < 2
            or not np.isfinite(time_s).all() or not np.isfinite(ratio).all()
            or not np.isfinite(target) or np.any(np.diff(time_s) <= 0)):
        raise ValueError("Interpolation requires finite matching series and increasing times.")
    if ratio[0] >= target:
        raise ValueError("No below-threshold starting sample is available to bracket a crossing.")
    brackets = np.flatnonzero((ratio[:-1] < target) & (ratio[1:] >= target))
    if not len(brackets):
        raise ValueError(f"No crossing of {target:g} is available.")
    left = int(brackets[0])
    fraction = float((target - ratio[left]) / (ratio[left + 1] - ratio[left]))
    time = float(time_s[left] + fraction * (time_s[left + 1] - time_s[left]))
    return Crossing(time, left, left + 1, fraction)


def loading_metrics(flow_ml_min: float, t10_s: float) -> dict:
    """Approved column-only hold-up correction; reporting boundary is mL and mg.

    V_H = V_bed*[epsilon_c+(1-epsilon_c)*epsilon_p]. This assumes all pore
    liquid is accessible and excludes tubing, mixers, and detector volumes.
    DBC10 = C_F*(Q*t10 - V_H)/V_bed, per geometric packed-bed volume.
    It is the prescribed feed-volume correction, not an effluent-integral capacity.
    """
    if not np.isfinite([flow_ml_min, t10_s]).all() or flow_ml_min <= 0 or t10_s <= 0:
        raise ValueError("Flow and breakthrough time must be finite and positive.")
    bed = float(np.pi * COLUMN_DIAMETER_M**2 * COLUMN_LENGTH_M / 4)
    hold_up = bed * (BED_POROSITY + (1 - BED_POROSITY) * PARTICLE_POROSITY)
    flow = flow_ml_min * M3_PER_ML / SECONDS_PER_MINUTE
    feed_mass_mg_ml = FEED_CONCENTRATION_MOL_M3 * MOLAR_MASS_KG_MOL * MG_PER_KG * M3_PER_ML
    return {
        "flow_ml_min": float(flow_ml_min), "t10_s": float(t10_s),
        "bed_volume_ml": bed / M3_PER_ML,
        "hold_up_volume_ml": hold_up / M3_PER_ML,
        "loaded_feed_volume_ml": flow * t10_s / M3_PER_ML,
        "nominal_residence_time_s": bed / flow,
        "feed_concentration_mg_ml": feed_mass_mg_ml,
        "dbc10_mg_ml_bed": feed_mass_mg_ml * (flow * t10_s - hold_up) / bed,
    }


def mass_trajectories(result: SimulationResult) -> dict[str, np.ndarray]:
    """Protein masses in mg, without altering raw states; FV cells use volume weights.

    Cumulative fluxes use the trapezoidal rule on raw output samples:
    https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.cumulative_trapezoid.html
    Bulk liquid, particle-pore liquid, and particle solid volumes are distinct.
    """
    mass_per_mol = MOLAR_MASS_KG_MOL * MG_PER_KG
    cell_factor = result.cell_bed_volume_m3 * mass_per_mol
    return {
        "cumulative_inlet_mass_mg": result.flow_m3_s * mass_per_mol * cumulative_trapezoid(
            result.inlet_mol_m3, result.time_s, initial=0),
        "cumulative_outlet_mass_mg": result.flow_m3_s * mass_per_mol * cumulative_trapezoid(
            result.outlet_mol_m3, result.time_s, initial=0),
        "liquid_mass_mg": cell_factor * (
            BED_POROSITY * result.bulk_liquid_mol_m3.sum(axis=1)
            + (1 - BED_POROSITY) * PARTICLE_POROSITY * result.pore_liquid_mol_m3.sum(axis=1)),
        "adsorbed_mass_mg": cell_factor * (1 - BED_POROSITY) * (1 - PARTICLE_POROSITY)
            * result.adsorbed_mol_m3_solid.sum(axis=1),
    }


def cumulative_mass_at_crossing(
    time_s: np.ndarray, mass_rate_mg_s: np.ndarray, cumulative_mass_mg: np.ndarray, crossing: Crossing,
) -> float:
    """Integrate the partial final interval exactly for linearly interpolated flux.

    Interpolating a cumulative integral itself would omit the quadratic term.
    """
    left, right, fraction = crossing.left, crossing.right, crossing.fraction
    dt = fraction * (time_s[right] - time_s[left])
    rate_at_crossing = mass_rate_mg_s[left] + fraction * (mass_rate_mg_s[right] - mass_rate_mg_s[left])
    return float(cumulative_mass_mg[left] + 0.5 * dt * (mass_rate_mg_s[left] + rate_at_crossing))


def normalized_mass_residual(inlet_mg: float, outlet_mg: float, liquid_mg: float, adsorbed_mg: float) -> float:
    """Signed (mass in - mass out - column inventory)/mass in, dimensionless."""
    if not np.isfinite([inlet_mg, outlet_mg, liquid_mg, adsorbed_mg]).all() or inlet_mg <= 0:
        raise ValueError("Mass residual requires finite masses and positive cumulative inlet mass.")
    return float((inlet_mg - outlet_mg - liquid_mg - adsorbed_mg) / inlet_mg)


def check_mass_acceptance(t10_residual: float, end_residual: float) -> None:
    if not np.isfinite([t10_residual, end_residual]).all() or max(abs(t10_residual), abs(end_residual)) > ACCEPTANCE_LIMIT:
        raise ValueError("Mass-balance residual exceeds 1.0% of cumulative inlet mass.")


def analyze_result(result: SimulationResult) -> tuple[dict, dict[str, np.ndarray]]:
    """Derive all quantities from raw solver data; no metric override inputs exist."""
    drawdown = validate_raw_curve(result)
    crossing = first_crossing(result.time_s, result.outlet_ratio)
    metrics = loading_metrics(result.flow_ml_min, crossing.time_s)
    masses = mass_trajectories(result)
    factor = result.flow_m3_s * MOLAR_MASS_KG_MOL * MG_PER_KG
    at_t10 = {
        "inlet": cumulative_mass_at_crossing(result.time_s, factor * result.inlet_mol_m3,
                                              masses["cumulative_inlet_mass_mg"], crossing),
        "outlet": cumulative_mass_at_crossing(result.time_s, factor * result.outlet_mol_m3,
                                               masses["cumulative_outlet_mass_mg"], crossing),
        "liquid": float(np.interp(crossing.time_s, result.time_s, masses["liquid_mass_mg"])),
        "adsorbed": float(np.interp(crossing.time_s, result.time_s, masses["adsorbed_mass_mg"])),
    }
    at_end = {name: float(masses[key][-1]) for name, key in (
        ("inlet", "cumulative_inlet_mass_mg"), ("outlet", "cumulative_outlet_mass_mg"),
        ("liquid", "liquid_mass_mg"), ("adsorbed", "adsorbed_mass_mg"))}
    for label, snapshot in (("t10", at_t10), ("end", at_end)):
        metrics.update({f"{phase}_mass_{label}_mg": mass for phase, mass in snapshot.items()})
        metrics[f"mass_balance_residual_{label}_fraction"] = normalized_mass_residual(*snapshot.values())
    metrics.update({
        "t10_left_sample_s": float(result.time_s[crossing.left]),
        "t10_right_sample_s": float(result.time_s[crossing.right]),
        "t10_interpolation_fraction": crossing.fraction,
        "max_raw_ratio_drawdown": drawdown, "final_outlet_inlet_ratio": float(result.outlet_ratio[-1]),
        "raw_ratio_numerical_tolerance": raw_ratio_tolerance(result.atol),
        "simulation_end_s": float(result.time_s[-1]), "load_extensions": result.load_extensions,
        "selected_samples_per_residence_time": result.samples_per_residence_time,
        "selected_axial_cells": result.axial_cells, "selected_rtol": result.rtol,
        "selected_atol_mol_m3": result.atol,
    })
    return metrics, masses


def select_numerical_level(metrics: np.ndarray) -> tuple[int, bool, str]:
    """Select from (three levels, three flows, t10/DBC) relative to the finest.

    First compare middle versus fine at every flow for BOTH metrics. Only
    after that gate passes may a coarser level be selected against the fine
    reference. Relative deltas always use abs(finest metric) as denominator.
    """
    metrics = np.asarray(metrics, dtype=float)
    if metrics.shape != (3, 3, 2) or not np.isfinite(metrics).all() or np.any(metrics <= 0):
        raise ValueError("Refinement requires positive finite t10 and DBC at all levels and flows.")
    errors = abs(metrics - metrics[-1]) / abs(metrics[-1])
    converged = bool(np.all(errors[-2] <= ACCEPTANCE_LIMIT))
    if not converged:
        return 2, False, "Unresolved numerical sensitivity: middle-to-finest change exceeds 1.0%; finest prescribed level selected."
    selected = int(np.flatnonzero(np.all(errors <= ACCEPTANCE_LIMIT, axis=(1, 2)))[0])
    return selected, True, "none"


def validate_refinement(report: RefinementResult) -> None:
    expected = select_numerical_level(report.metrics)
    if (report.selected_index, report.converged, report.disclosure) != expected:
        raise ValueError("Refinement selection or unresolved-sensitivity disclosure is inconsistent.")


def run_refinement() -> tuple[NumericalSettings, list[RefinementResult]]:
    """Exactly nine runs per factor, in order, retaining previously selected settings."""
    current = NumericalSettings()
    reports = []
    for factor in REFINEMENT_FACTORS:
        if factor == "sampling":
            trials = tuple(replace(current, samples_per_residence_time=level) for level in OUTPUT_SAMPLING_LEVELS)
        elif factor == "axial_grid":
            trials = tuple(replace(current, axial_cells=level) for level in AXIAL_GRID_LEVELS)
        else:
            trials = tuple(replace(current, rtol=rtol, atol=atol) for rtol, atol in SOLVER_TOLERANCE_LEVELS)
        values = np.empty((3, 3, 2))
        for level_index, numerical in enumerate(trials):
            for flow_index, flow in enumerate(FLOW_RATES_ML_MIN):
                result = simulate_flow(flow, **asdict(numerical))
                metrics, _ = analyze_result(result)
                values[level_index, flow_index] = (metrics["t10_s"], metrics["dbc10_mg_ml_bed"])
                print(f"REFINE {factor} level {level_index + 1}/3, Q={flow:g}: "
                      f"t10={metrics['t10_s']:.6f} s, DBC10={metrics['dbc10_mg_ml_bed']:.6f} mg/mL bed", flush=True)
        selected, converged, disclosure = select_numerical_level(values)
        report = RefinementResult(factor, trials, values, selected, converged, disclosure)
        validate_refinement(report)
        reports.append(report)
        current = trials[selected]
        print(f"SELECT {factor}: {current}; converged={converged}; disclosure={disclosure}", flush=True)
    return current, reports


def make_output_tables(
    production: list[SimulationResult], reports: list[RefinementResult],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, curves = [], []
    for flow_index, result in enumerate(production):
        row, masses = analyze_result(result)
        row.update({"data_origin": "simulated", "random_seed": SEED,
                    "cadet_process_version": version("CADET-Process"), "cadet_core_version": version("CADET-Core")})
        for report in reports:
            errors = abs(report.metrics - report.metrics[-1]) / abs(report.metrics[-1])
            prefix = report.factor
            row.update({
                f"{prefix}_t10_relative_change_fraction": errors[-2, flow_index, 0],
                f"{prefix}_dbc10_relative_change_fraction": errors[-2, flow_index, 1],
                f"{prefix}_selected_t10_error_fraction": errors[report.selected_index, flow_index, 0],
                f"{prefix}_selected_dbc10_error_fraction": errors[report.selected_index, flow_index, 1],
                f"{prefix}_selected_level_index": report.selected_index,
                f"{prefix}_converged": report.converged, f"{prefix}_disclosure": report.disclosure,
            })
            for index, label in enumerate(("coarse", "middle", "fine")):
                row[f"{prefix}_{label}_t10_s"] = report.metrics[index, flow_index, 0]
                row[f"{prefix}_{label}_dbc10_mg_ml_bed"] = report.metrics[index, flow_index, 1]
        rows.append(row)
        curves.append(pd.DataFrame({
            "data_origin": "simulated", "flow_ml_min": result.flow_ml_min,
            "time_s": result.time_s, "inlet_concentration_mol_m3": result.inlet_mol_m3,
            "outlet_concentration_mol_m3": result.outlet_mol_m3,
            "outlet_inlet_ratio": result.outlet_ratio, **masses,
        }))
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


CURVE_COLUMNS = (
    "data_origin", "flow_ml_min", "time_s", "inlet_concentration_mol_m3",
    "outlet_concentration_mol_m3", "outlet_inlet_ratio", "cumulative_inlet_mass_mg",
    "cumulative_outlet_mass_mg", "liquid_mass_mg", "adsorbed_mass_mg",
)
ACCEPTANCE_COLUMNS = (
    "curve_completion_passed", "interpolation_passed", "volumes_units_passed",
    "mass_balance_passed", "numerical_acceptance_passed", "raw_output_preserved", "acceptance_passed",
)
SUMMARY_COLUMNS = (
    "flow_ml_min", "t10_s", "bed_volume_ml", "hold_up_volume_ml", "loaded_feed_volume_ml",
    "nominal_residence_time_s", "feed_concentration_mg_ml", "dbc10_mg_ml_bed",
    "mass_balance_residual_t10_fraction", "mass_balance_residual_end_fraction",
    "selected_samples_per_residence_time", "selected_axial_cells", "selected_rtol", "selected_atol_mol_m3",
    "t10_left_sample_s", "t10_right_sample_s", "t10_interpolation_fraction", "data_origin", "random_seed",
    "raw_ratio_numerical_tolerance",
)


def require_close(actual, expected, label: str) -> None:
    if not np.allclose(actual, expected, rtol=5e-12, atol=1e-12):
        raise ValueError(f"Inconsistent {label}; quantities must be calculated without overrides.")


def validate_output_tables(
    parameters: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame,
    *, require_acceptance_flags: bool = True,
) -> None:
    """Check exported quantities against raw curves, equations, and the refinement ledger.

    Fractions are dimensionless (0.01 = 1.0%). Tolerances in this checker only
    accommodate CSV float round trips; they do not relax the study criteria.
    """
    required_summary = set(SUMMARY_COLUMNS) | (set(ACCEPTANCE_COLUMNS) if require_acceptance_flags else set())
    if not required_summary <= set(summary) or not set(CURVE_COLUMNS) <= set(curves):
        raise ValueError("Required unit-bearing result columns are missing.")
    if not set(PARAMETER_RECORDS.columns) <= set(parameters):
        raise ValueError("Parameter provenance columns are missing.")
    for name, table in (("parameters", parameters), ("summary", summary), ("curves", curves)):
        if table.empty or table.isna().any().any() or not np.isfinite(table.select_dtypes(include="number")).all().all():
            raise ValueError(f"Empty, missing, or nonfinite data in {name}.")
    if not set(parameters.provenance) <= {"sourced", "adapted", "assumed", "derived"}:
        raise ValueError("Invalid parameter provenance status.")
    for name in ("name", "symbol", "unit", "cadet_mapping", "source_or_rationale"):
        if not parameters[name].astype(str).str.strip().ne("").all():
            raise ValueError(f"Empty parameter field: {name}.")
    if len(summary) != 3 or set(summary.flow_ml_min) != set(FLOW_RATES_ML_MIN) or set(curves.flow_ml_min) != set(FLOW_RATES_ML_MIN):
        raise ValueError("Results must cover exactly the three prescribed flow rates.")
    if not summary.data_origin.eq("simulated").all() or not curves.data_origin.eq("simulated").all():
        raise ValueError("All results must be identified as simulated.")
    if not summary.random_seed.eq(SEED).all():
        raise ValueError("Incorrect project random seed.")
    if require_acceptance_flags and not summary[list(ACCEPTANCE_COLUMNS)].eq(True).all().all():
        raise ValueError("A programmatic acceptance check failed.")
    ordered = summary.set_index("flow_ml_min").loc[list(FLOW_RATES_ML_MIN)]
    for name in ("selected_samples_per_residence_time", "selected_axial_cells", "selected_rtol", "selected_atol_mol_m3"):
        if summary[name].nunique() != 1:
            raise ValueError("Production numerical settings must agree across flows.")
    for flow, row in ordered.iterrows():
        group = curves.loc[curves.flow_ml_min.eq(flow)]
        time = group.time_s.to_numpy()
        inlet = group.inlet_concentration_mol_m3.to_numpy()
        outlet = group.outlet_concentration_mol_m3.to_numpy()
        ratio = group.outlet_inlet_ratio.to_numpy()
        if len(time) < 2 or time[0] != 0 or np.any(np.diff(time) <= 0):
            raise ValueError("Curve times must start at zero and strictly increase.")
        require_close(inlet, FEED_CONCENTRATION_MOL_M3, "constant inlet concentration")
        require_close(ratio, outlet / inlet, "raw outlet/inlet ratio")
        drawdown = float(np.max(np.maximum.accumulate(ratio) - ratio))
        tolerance = raw_ratio_tolerance(row.selected_atol_mol_m3)
        require_close(row.raw_ratio_numerical_tolerance, tolerance, "raw ratio numerical tolerance")
        if drawdown > tolerance or ratio.min() < -tolerance or ratio.max() > 1 + tolerance:
            raise ValueError("Raw curve is not nondecreasing/bounded within numerical tolerance.")
        if ratio[-1] < TARGET_OUTLET_RATIO:
            raise ValueError("Raw curve has not reached 0.95.")
        crossing = first_crossing(time, ratio)
        expected = loading_metrics(flow, crossing.time_s)
        for name, value in expected.items():
            if name != "flow_ml_min":
                require_close(row[name], value, name)
        if row.dbc10_mg_ml_bed <= 0:
            raise ValueError("The selected hypothetical loading must have positive DBC10.")
        require_close(row.t10_left_sample_s, time[crossing.left], "left interpolation bracket")
        require_close(row.t10_right_sample_s, time[crossing.right], "right interpolation bracket")
        require_close(row.t10_interpolation_fraction, crossing.fraction, "interpolation fraction")
        require_close(row.max_raw_ratio_drawdown, drawdown, "raw curve drawdown")
        require_close(row.final_outlet_inlet_ratio, ratio[-1], "final outlet ratio")
        require_close(row.simulation_end_s, time[-1], "simulation end")
        dt = row.nominal_residence_time_s / row.selected_samples_per_residence_time
        require_close(np.diff(time), dt, "samples per nominal residence time")
        flow_si = flow * M3_PER_ML / SECONDS_PER_MINUTE
        mass_factor = flow_si * MOLAR_MASS_KG_MOL * MG_PER_KG
        snapshots = {"t10": {}, "end": {}}
        for phase, concentration in (("inlet", inlet), ("outlet", outlet)):
            cumulative = cumulative_trapezoid(concentration, time, initial=0) * mass_factor
            require_close(group[f"cumulative_{phase}_mass_mg"], cumulative, f"cumulative {phase} mass")
            snapshots["t10"][phase] = cumulative_mass_at_crossing(time, concentration * mass_factor, cumulative, crossing)
            snapshots["end"][phase] = float(cumulative[-1])
        for phase in ("liquid", "adsorbed"):
            values = group[f"{phase}_mass_mg"].to_numpy()
            require_close(values[0], 0.0, f"initial {phase} inventory")
            snapshots["t10"][phase] = float(np.interp(crossing.time_s, time, values))
            snapshots["end"][phase] = float(values[-1])
        for label, masses in snapshots.items():
            for phase, value in masses.items():
                require_close(row[f"{phase}_mass_{label}_mg"], value, f"{phase} mass at {label}")
            residual = normalized_mass_residual(*masses.values())
            require_close(row[f"mass_balance_residual_{label}_fraction"], residual, f"mass residual at {label}")
        check_mass_acceptance(row.mass_balance_residual_t10_fraction, row.mass_balance_residual_end_fraction)
    for factor in REFINEMENT_FACTORS:
        try:
            metrics = np.stack([ordered[[f"{factor}_{label}_t10_s", f"{factor}_{label}_dbc10_mg_ml_bed"]].to_numpy()
                                for label in ("coarse", "middle", "fine")])
            selected, converged, disclosure = select_numerical_level(metrics)
            errors = abs(metrics - metrics[-1]) / abs(metrics[-1])
            if (not ordered[f"{factor}_converged"].eq(converged).all()
                    or not ordered[f"{factor}_disclosure"].eq(disclosure).all()
                    or not ordered[f"{factor}_selected_level_index"].eq(selected).all()):
                raise ValueError("Convergence/selection/disclosure does not match tested levels.")
            for metric_index, metric in enumerate(("t10", "dbc10")):
                require_close(ordered[f"{factor}_{metric}_relative_change_fraction"], errors[-2, :, metric_index], "finest-level convergence delta")
                require_close(ordered[f"{factor}_selected_{metric}_error_fraction"], errors[selected, :, metric_index], "selected-to-finest delta")
            if factor == "sampling":
                require_close(ordered.selected_samples_per_residence_time, OUTPUT_SAMPLING_LEVELS[selected], "selected sampling")
            elif factor == "axial_grid":
                require_close(ordered.selected_axial_cells, AXIAL_GRID_LEVELS[selected], "selected grid")
            else:
                require_close(ordered.selected_rtol, SOLVER_TOLERANCE_LEVELS[selected][0], "selected rtol")
                require_close(ordered.selected_atol_mol_m3, SOLVER_TOLERANCE_LEVELS[selected][1], "selected atol")
        except KeyError as error:
            raise ValueError("Required refinement ledger columns are missing.") from error


def parameter_table(selected: NumericalSettings) -> pd.DataFrame:
    """Physical provenance plus numerical levels, baselines, selections, and seed."""
    rows = []
    for index, level in enumerate(OUTPUT_SAMPLING_LEVELS):
        rows.append((f"Output sampling level {index + 1}", "n_sample", level, "samples/residence time",
                     "simulator.time_resolution = t_R/n_sample", "sourced", "Approved brief: Numerical Refinement Hyperparameters."))
    for index, level in enumerate(AXIAL_GRID_LEVELS):
        rows.append((f"Axial grid level {index + 1}", "n_col", level, "cells", "column.discretization.ncol / NCOL",
                     "sourced", "Approved brief: Numerical Refinement Hyperparameters."))
    for index, (rtol, atol) in enumerate(SOLVER_TOLERANCE_LEVELS):
        rows.extend([
            (f"Relative tolerance level {index + 1}", "rtol", rtol, "1", "time_integrator_parameters.reltol / RELTOL",
             "sourced", "Approved brief: Numerical Refinement Hyperparameters."),
            (f"Absolute tolerance level {index + 1}", "atol", atol, "mol/m^3", "time_integrator_parameters.abstol / ABSTOL",
             "sourced", "Approved brief, applied in native molar concentration units."),
        ])
    for label, numerical, provenance in (("Baseline", NumericalSettings(), "sourced"), ("Selected", selected, "derived")):
        for name, value in asdict(numerical).items():
            unit = {"samples_per_residence_time": "samples/residence time", "axial_cells": "cells", "rtol": "1", "atol": "mol/m^3"}[name]
            rows.append((f"{label} {name}", name, value, unit, {
                "samples_per_residence_time": "simulator.time_resolution = t_R/n_sample",
                "axial_cells": "column.discretization.ncol / NCOL", "rtol": "time_integrator_parameters.reltol / RELTOL",
                "atol": "time_integrator_parameters.abstol / ABSTOL"}[name], provenance,
                "Approved brief: prescribed baseline." if label == "Baseline" else "Sequential one-factor selection against the finest tested level for every flow and both metrics."))
    rows.append(("Project random seed", "seed", SEED, "1", "random.seed and numpy.random.seed", "assumed", "Fixed user-prescribed project seed 20261025."))
    rows.extend([
        ("Raw ratio tolerance floor", "delta_ratio_min", RAW_MONOTONICITY_ATOL, "1", "raw_ratio_tolerance",
         "assumed", "Declared numerical screening floor for finite-precision raw ratios; no data correction."),
        ("Raw ratio absolute-tolerance multiplier", "f_atol", RAW_RATIO_ATOL_MULTIPLIER, "1", "raw_ratio_tolerance",
         "assumed", "Allowance for comparing two approximate concentrations; not a rigorous solver error bound."),
        ("Selected raw ratio tolerance", "delta_ratio", raw_ratio_tolerance(selected.atol), "1", "raw_ratio_tolerance",
         "derived", "max(1e-6, 4*selected_atol/C_F); raw output is never smoothed or clipped."),
    ])
    return pd.concat([PARAMETER_RECORDS, pd.DataFrame.from_records(rows, columns=PARAMETER_RECORDS.columns)], ignore_index=True)


def validate_production(
    production: list[SimulationResult], reports: list[RefinementResult], selected: NumericalSettings,
    parameters: pd.DataFrame, summary: pd.DataFrame, curves: pd.DataFrame,
) -> None:
    """Require a sequential ledger and tables recomputed directly from raw production states."""
    if tuple(report.factor for report in reports) != REFINEMENT_FACTORS:
        raise ValueError("Refinement must be sampling, axial grid, then tolerances.")
    current = NumericalSettings()
    for report in reports:
        validate_refinement(report)
        if report.factor == "sampling":
            expected = tuple(replace(current, samples_per_residence_time=level) for level in OUTPUT_SAMPLING_LEVELS)
        elif report.factor == "axial_grid":
            expected = tuple(replace(current, axial_cells=level) for level in AXIAL_GRID_LEVELS)
        else:
            expected = tuple(replace(current, rtol=r, atol=a) for r, a in SOLVER_TOLERANCE_LEVELS)
        if report.settings != expected:
            raise ValueError("A non-refined numerical factor changed during a refinement stage.")
        current = report.settings[report.selected_index]
    if current != selected or tuple(result.flow_ml_min for result in production) != FLOW_RATES_ML_MIN:
        raise ValueError("Production flow coverage or selected settings are inconsistent.")
    for result in production:
        if (result.samples_per_residence_time, result.axial_cells, result.rtol, result.atol) != (
                selected.samples_per_residence_time, selected.axial_cells, selected.rtol, selected.atol):
            raise ValueError("Production did not use the selected settings.")
        require_close(result.cell_bed_volume_m3 * result.axial_cells, BED_VOLUME_M3, "FV bed volume")
    expected_summary, expected_curves = make_output_tables(production, reports)
    if not summary.equals(expected_summary) or not curves.equals(expected_curves) or not parameters.equals(parameter_table(selected)):
        raise ValueError("Manual result/parameter overrides detected; tables must come directly from raw states.")
    validate_output_tables(parameters, summary, curves, require_acceptance_flags=False)
    for name in ACCEPTANCE_COLUMNS:
        summary[name] = True  # only reached after all independent acceptance checks pass
    validate_output_tables(parameters, summary, curves)


def render_figures(summary: pd.DataFrame, curves: pd.DataFrame) -> dict[str, bytes]:
    """Render raw samples and calculated metrics in memory with simulation labels."""
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    colors = ("#0072B2", "#D55E00", "#009E73")
    images = {}
    style = {"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
             "axes.spines.right": False, "axes.titlesize": 12, "axes.labelsize": 11}
    with rc_context(style):
        figure = Figure(figsize=(7.6, 4.8), dpi=300)
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        for color, flow in zip(colors, FLOW_RATES_ML_MIN):
            group = curves.loc[curves.flow_ml_min.eq(flow)]
            ax.plot(group.time_s / SECONDS_PER_MINUTE, group.outlet_inlet_ratio,
                    color=color, linewidth=2, label=f"{flow:g} mL/min")
        for target in (0.10, 0.95):
            ax.axhline(target, color="#777777", linewidth=0.9, linestyle="--")
            label_y = target - 0.045 if target > 0.5 else target + 0.02
            ax.text(0.99, label_y, f"{target:.2f}", transform=ax.get_yaxis_transform(), ha="right", fontsize=9)
        ax.set(xlabel="Loading time (min)", ylabel=r"Outlet / inlet concentration, $C_{out}/C_F$ (1)",
               title="Simulated protein breakthrough at three flow rates", ylim=(-0.02, 1.08), xlim=(0, None))
        ax.legend(title="Constant flow", frameon=False, loc="center right", bbox_to_anchor=(0.99, 0.44))
        ax.grid(axis="both", alpha=0.18)
        figure.text(0.5, 0.035, "Simulated CADET predictions for one hypothetical protein\nFixed column, feed, binding, dispersion and film transfer; no experimental validation",
                    ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(left=0.12, right=0.98, bottom=0.23, top=0.88)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=300)
        images[RESULT_FILENAMES[3]] = buffer.getvalue()

        figure = Figure(figsize=(8.2, 4.3), dpi=300)
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        ordered = summary.sort_values("flow_ml_min")
        for ax, column, divisor, ylabel, color, label in (
            (axes[0], "t10_s", SECONDS_PER_MINUTE, "10% breakthrough time (min)", colors[0], r"Simulated $t_{10}$"),
            (axes[1], "dbc10_mg_ml_bed", 1, "DBC at 10% breakthrough (mg/mL bed)", colors[1], r"Simulated $DBC_{10}$"),
        ):
            # Discrete predictions only; no interpolated trend between flow rates.
            ax.scatter(ordered.flow_ml_min, ordered[column] / divisor, s=62, color=color, label=label)
            ax.set(xlabel="Flow rate (mL/min)", ylabel=ylabel, xticks=FLOW_RATES_ML_MIN)
            ax.set_ylim(0, 1.18 * float(ordered[column].max()) / divisor)
            ax.grid(alpha=0.18)
            ax.legend(frameon=False, fontsize=10)
        figure.suptitle("Simulated breakthrough time and binding capacity", fontsize=13, y=0.96)
        figure.text(0.5, 0.035, "Simulated predictions; hypothetical protein and adsorption parameters\nDBC uses the approved column-only hold-up correction per packed-bed volume",
                    ha="center", fontsize=9, color="#444444")
        figure.subplots_adjust(left=0.10, right=0.98, bottom=0.26, top=0.84, wspace=0.36)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=300)
        images[RESULT_FILENAMES[4]] = buffer.getvalue()
    return images


def run_study() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    selected, reports = run_refinement()
    production = []
    for flow in FLOW_RATES_ML_MIN:
        print(f"PRODUCTION Q={flow:g} mL/min; {selected}", flush=True)
        production.append(simulate_flow(flow, **asdict(selected)))
    parameters = parameter_table(selected)
    summary, curves = make_output_tables(production, reports)
    validate_production(production, reports, selected, parameters, summary, curves)
    # Validate CSV round trips and render both figures in memory before creating
    # results/. Failed scientific checks leave no result files or staging files.
    outputs = {name: table.to_csv(index=False, lineterminator="\n").encode("utf-8")
               for name, table in zip(RESULT_FILENAMES[:3], (parameters, summary, curves))}
    validate_output_tables(*(pd.read_csv(io.BytesIO(outputs[name])) for name in RESULT_FILENAMES[:3]))
    outputs.update(render_figures(summary, curves))
    destination = PROJECT_DIR / "results"
    if destination.exists() and set(path.name for path in destination.iterdir()) - set(RESULT_FILENAMES):
        raise ValueError("Unexpected existing results files; refusing to mix output sets.")
    destination.mkdir(exist_ok=True)
    for name in RESULT_FILENAMES:
        (destination / name).write_bytes(outputs[name])
    print(summary[["flow_ml_min", "t10_s", "dbc10_mg_ml_bed", "mass_balance_residual_t10_fraction",
                   "mass_balance_residual_end_fraction", "acceptance_passed"]].to_string(index=False), flush=True)
    print("Complete study accepted; wrote exactly five required outputs to results/.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group()
    commands.add_argument("--smoke", action="store_true", help="check all three baseline simulated loads; save no results")
    commands.add_argument("--run", action="store_true", help="run sequential refinement, acceptance checks, and the complete study")
    args = parser.parse_args()
    if args.run:
        run_study()
        return
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
