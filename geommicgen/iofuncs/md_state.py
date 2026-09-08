"""
Module containing the state of a microstructure generation run.

The configuration a run arrives at is written as a microstructure file, which carries the
particles and nothing else. What the run recorded about how it got there -- the histories
the motion analysis plots, the number of steps taken, how long it took -- is numeric, and
is written beside it in a compressed array file. A generated microstructure can therefore
be analysed again without the generation method itself having to be reconstructed, which
is what reading back a pickle of it used to be for.
"""

import os

import numpy as np


STATE_FILE_NAME = "md_state.npz"


class ThermostatState:
    """
    Class for the state of the thermostat of a generation run.

    Attributes
    ----------
    temp_change_steps: list(int)
        Steps at which the thermostat changed the temperature.

    ratio: list(float)
        History of the overlap ratio the thermostat controlled.
    """

    def __init__(self, temp_change_steps, ratio):
        """Initizalizer for the ThermostatState class."""
        self.temp_change_steps = temp_change_steps
        self.ratio = ratio


class GenerationState:
    """
    Class for the state of a microstructure generation run read back from a file.

    It carries the attributes the post processing reads off a generation method, and no
    behaviour: a run that has already happened is a record, not something that can be
    continued.

    Attributes
    ----------
    step: int
        Number of steps the run took.

    time: float
        Duration of the run, in seconds. None when the run did not finish.

    status: bool
        Whether the run reached an admissible configuration.

    max_residue: float
        Overlap the run accepted as converged. None when the run did not start.

    total_overlap_history: list(float)
        Total overlap at every step.

    position_center_history: list(list(array))
        Position of the center of every particle, per particle and per recorded step.
        None when the run recorded no positions.

    kinetic_energy_history: list(float)
        Kinetic energy at every step.

    thermic_energy_history: list(float)
        Thermic energy at every step.

    all_dt: list(float)
        Time increment used at every step.

    thermostat: `.ThermostatState`
        State of the thermostat of the run.
    """

    def __init__(
        self,
        step,
        time,
        status,
        max_residue,
        total_overlap_history,
        position_center_history,
        kinetic_energy_history,
        thermic_energy_history,
        all_dt,
        thermostat,
    ):
        """Initizalizer for the GenerationState class."""
        self.step = step
        self.time = time
        self.status = status
        self.max_residue = max_residue
        self.total_overlap_history = total_overlap_history
        self.position_center_history = position_center_history
        self.kinetic_energy_history = kinetic_energy_history
        self.thermic_energy_history = thermic_energy_history
        self.all_dt = all_dt
        self.thermostat = thermostat


def save_md_state(sample_dir, mic_generator):
    """
    Write the state of a generation run beside its microstructure.

    Parameters
    ----------
    sample_dir: str
        Directory of the sample the run produced.

    mic_generator: `.MolecularDynamicsSimulation`
        Generation method whose state is to be written.

    Returns
    -------
    str
        Path of the file that was written.
    """
    thermostat = getattr(mic_generator, "thermostat", None)
    arrays = {
        "step": int(mic_generator.step),
        "time": _scalar(mic_generator.time),
        "status": bool(mic_generator.status),
        "max_residue": _scalar(mic_generator.max_residue),
        "total_overlap_history": _history(mic_generator.total_overlap_history),
        "kinetic_energy_history": _history(mic_generator.kinetic_energy_history),
        "thermic_energy_history": _history(mic_generator.thermic_energy_history),
        "all_dt": _history(mic_generator.all_dt),
        "temp_change_steps": _history(
            getattr(thermostat, "temp_change_steps", []), dtype=int
        ),
        "ratio": _history(getattr(thermostat, "ratio", [])),
    }
    positions = _position_history(mic_generator.position_center_history)
    if positions is not None:
        arrays["position_center_history"] = positions
    # A run that recorded no positions writes no positions, rather than an empty array
    # that would read back as a history of no particles

    file_path = os.path.join(sample_dir, STATE_FILE_NAME)
    np.savez(file_path, **arrays)
    # Not compressed: the histories are trajectory floats, which deflate by a few per
    # cent and cost some thirty times the write

    return file_path


def load_md_state(file_path):
    """
    Read the state of a generation run.

    Parameters
    ----------
    file_path: str
        Path of the file to be read.

    Returns
    -------
    `.GenerationState`
        State of the run, or None when the file does not exist.
    """
    if not os.path.exists(file_path):
        return None

    with np.load(file_path) as state:
        positions = None
        if "position_center_history" in state:
            positions = list(state["position_center_history"])
        # The motion analysis walks the history of one particle at a time, so it is
        # handed one entry per particle. Each is a row of the array rather than a list
        # of its own: indexing a row by step gives the same position back, and building
        # the lists costs an array per particle per step

        return GenerationState(
            step=int(state["step"]),
            time=_optional(state["time"]),
            status=bool(state["status"]),
            max_residue=_optional(state["max_residue"]),
            total_overlap_history=state["total_overlap_history"].tolist(),
            position_center_history=positions,
            kinetic_energy_history=state["kinetic_energy_history"].tolist(),
            thermic_energy_history=state["thermic_energy_history"].tolist(),
            all_dt=state["all_dt"].tolist(),
            thermostat=ThermostatState(
                temp_change_steps=state["temp_change_steps"].tolist(),
                ratio=state["ratio"].tolist(),
            ),
        )


def _scalar(value):
    """
    Turn a value that may not have been recorded into an array.

    Parameters
    ----------
    value: float
        Value to be written, or None when the run did not record it.

    Returns
    -------
    array
        Array holding the value, holding *nan* when there is none.
    """
    return np.array(np.nan if value is None else float(value))


def _optional(value):
    """
    Turn an array written by `._scalar` back into a value.

    Parameters
    ----------
    value: array
        Array read back from the file.

    Returns
    -------
    float
        The value, or None when the run did not record it.
    """
    return None if np.isnan(value) else float(value)


def _history(history, dtype=float):
    """
    Turn a history of scalars into an array.

    Parameters
    ----------
    history: list
        History recorded by the run.

    dtype: type
        Type of the values of the history.

    Returns
    -------
    array
        Array holding the history.
    """
    return np.asarray(history, dtype=dtype)


def _position_history(history):
    """
    Turn the recorded positions of the particles into a single array.

    Parameters
    ----------
    history: list(list(array))
        Position of the center of every particle, per particle and per recorded step.

    Returns
    -------
    array
        Array of shape *(n_particles, n_steps, dim)*, or None when nothing was recorded.
    """
    if not history:
        return None

    n_steps = min(len(i_particle_history) for i_particle_history in history)
    if n_steps == 0:
        return None
    # A run interrupted part way through a step leaves one particle with an entry the
    # others do not have, so the history is cut at the last step every particle reached

    return np.asarray(
        [i_particle_history[:n_steps] for i_particle_history in history], dtype=float
    )
