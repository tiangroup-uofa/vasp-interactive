"""Process and scheduler controllers used by :class:`VaspInteractive`.

MPI process discovery is an optional capability because it depends on psutil.
Slurm step control intentionally uses the Slurm command-line tools instead of
psutil, since signals sent to a local ``srun`` process are not guaranteed to
reach all ranks in a job step.
"""

from abc import ABC, abstractmethod
import os
import shlex
import signal
from warnings import warn

from . import utils


class ControllerUnavailable(RuntimeError):
    """Raised when a process controller cannot locate its target."""


class ProcessController(ABC):
    """Common interface for local, MPI, and scheduler process control."""

    kind = "process"

    @property
    def match(self):
        """Compatibility information used by older callers."""
        return None

    @abstractmethod
    def send_signal(self, sig):
        """Send *sig* and return ``True`` only when it was dispatched."""


def _command_contains(command, executable):
    if not command:
        return False
    if isinstance(command, (list, tuple)):
        tokens = [str(token) for token in command]
    else:
        try:
            tokens = shlex.split(str(command))
        except ValueError:
            tokens = str(command).split()
    return any(os.path.basename(token) == executable for token in tokens)


def terminate_process_tree(pid, process=None):
    """Best-effort termination of a Popen process and all descendants.

    With psutil installed, descendants are killed before the root process. If
    psutil is unavailable, only the Popen/root process can be killed safely;
    callers should not treat that as MPI cleanup success.
    """
    psutil = utils.psutil
    if psutil is None:
        try:
            if process is not None:
                process.kill()
            else:
                os.kill(pid, signal.SIGKILL)
            return True
        except (OSError, AttributeError):
            return False

    try:
        root = psutil.Process(pid)
        descendants = root.children(recursive=True)
    except Exception:
        return False

    processes = list(reversed(descendants)) + [root]
    for proc in processes:
        try:
            proc.kill()
        except Exception:
            pass

    try:
        _, alive = psutil.wait_procs(processes, timeout=5)
    except Exception:
        return False
    return not alive


class MPIProcessController(ProcessController):
    """Control an MPI launcher discovered below the Popen process."""

    kind = "mpi"

    def __init__(
        self,
        root_pid,
        launcher,
        match,
        mpi_program="mpirun",
        vasp_program="vasp_std",
    ):
        self.root_pid = root_pid
        self.launcher = launcher
        self._match = match
        self.mpi_program = mpi_program
        self.vasp_program = vasp_program

    @property
    def match(self):
        return self._match

    @classmethod
    def discover(cls, root_pid, mpi_program="mpirun", vasp_program="vasp_std"):
        if utils.psutil is None:
            return None
        match = utils._find_mpi_process(
            root_pid, mpi_program=mpi_program, vasp_program=vasp_program
        )
        if match["type"] != "mpi" or match["process"] is None:
            return None
        return cls(
            root_pid,
            match["process"],
            match,
            mpi_program=mpi_program,
            vasp_program=vasp_program,
        )

    def _refresh(self):
        refreshed = self.discover(
            self.root_pid,
            mpi_program=self.mpi_program,
            vasp_program=self.vasp_program,
        )
        if refreshed is None:
            return False
        self.launcher = refreshed.launcher
        self._match = refreshed.match
        return True

    def send_signal(self, sig):
        # Killing only the launcher can leave MPI ranks orphaned. Terminate the
        # complete process tree for the force-kill path instead.
        if sig in (signal.SIGKILL, signal.SIGTERM):
            return terminate_process_tree(self.root_pid)

        for attempt in range(2):
            try:
                self.launcher.send_signal(sig)
                return True
            except Exception:
                if attempt == 0 and self._refresh():
                    continue
                return False
        return False


class SchedulerStepController(ProcessController, ABC):
    """Template for schedulers that signal a job step rather than a PID.

    This is intentionally an extension point. Only the Slurm implementation is
    provided and documented by this project.
    """

    kind = "scheduler"

    def __init__(self, step_id=None):
        self.step_id = step_id

    @property
    def match(self):
        return {"type": self.kind, "process": self.step_id}

    @abstractmethod
    def locate_step(self):
        """Return the scheduler step identifier."""

    @abstractmethod
    def signal_step(self, step_id, sig):
        """Signal a scheduler step."""

    def send_signal(self, sig):
        if self.step_id is None:
            self.step_id = self.locate_step()
        if self.step_id is None:
            raise ControllerUnavailable("Cannot locate the scheduler job step")

        try:
            self.signal_step(self.step_id, sig)
        except Exception:
            # A step can disappear or be replaced while VASP is restarted.
            self.step_id = self.locate_step()
            if self.step_id is None:
                raise ControllerUnavailable("Cannot refresh the scheduler job step")
            self.signal_step(self.step_id, sig)
        return True


class SlurmStepController(SchedulerStepController):
    """Pause and resume an ``srun`` job step using ``squeue``/``scancel``."""

    kind = "slurm"

    def __init__(self, step_id=None, vasp_program="vasp_std"):
        super().__init__(step_id=step_id)
        self.vasp_program = vasp_program

    def locate_step(self):
        return utils._locate_slurm_step(vasp_program=self.vasp_program)

    def signal_step(self, step_id, sig):
        utils._slurm_signal(step_id, sig)


def get_process_controller(
    root_pid, command=None, mpi_program="mpirun", vasp_program="vasp_std"
):
    """Select a controller for a newly started VASP process.

    Explicit ``srun`` commands are recognized without psutil. For wrapper
    scripts, the psutil-backed process-tree discovery remains available when
    psutil is installed.
    """
    if _command_contains(command, "srun"):
        if utils._get_slurm_jobid() is not None:
            return SlurmStepController(vasp_program=vasp_program)
        warn("srun was requested, but no Slurm job ID is available; pausing is disabled.")
        return None

    match = utils._find_mpi_process(
        root_pid, mpi_program=mpi_program, vasp_program=vasp_program
    )
    if match["type"] == "slurm":
        return SlurmStepController(
            step_id=match["process"], vasp_program=vasp_program
        )
    if match["type"] == "mpi" and match["process"] is not None:
        return MPIProcessController(
            root_pid,
            match["process"],
            match,
            mpi_program=mpi_program,
            vasp_program=vasp_program,
        )
    return None
