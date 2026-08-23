import pytest

from vasp_interactive import process_control, utils
from vasp_interactive.vasp_interactive import VaspInteractive


class FakeProcess:
    _next_pid = 1000

    def __init__(self, name, children=()):
        self._name = name
        self._children = list(children)
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1

    def name(self):
        return self._name

    def send_signal(self, sig):
        if not hasattr(self, "signals"):
            self.signals = []
        self.signals.append(sig)

    def children(self, recursive=False):
        if not recursive:
            return list(self._children)
        descendants = []
        for child in self._children:
            descendants.append(child)
            descendants.extend(child.children(recursive=True))
        return descendants


def install_process_tree(monkeypatch, launcher, vasp_name="vasp_std"):
    vasp = FakeProcess(vasp_name)
    launcher_process = FakeProcess(launcher, [vasp])
    root = FakeProcess("shell", [launcher_process])
    monkeypatch.setattr(utils.psutil, "Process", lambda pid: root)
    return root, launcher_process


@pytest.mark.parametrize(
    "launcher",
    ["mpirun", "mpiexec", "orterun", "prterun", "prte", "oshrun", "shmemrun"],
)
@pytest.mark.parametrize("vasp_name", ["vasp_std", "vasp.6.4.3_std"])
def test_supported_mpi_process_matrix(monkeypatch, launcher, vasp_name):
    root, launcher_process = install_process_tree(monkeypatch, launcher, vasp_name)

    match = utils._find_mpi_process(root.pid)

    assert match["type"] == "mpi"
    assert match["process"] is launcher_process


def test_custom_mpi_launcher(monkeypatch):
    root, launcher_process = install_process_tree(monkeypatch, "site-mpirun")

    match = utils._find_mpi_process(root.pid, mpi_program="site-mpirun")

    assert match["type"] == "mpi"
    assert match["process"] is launcher_process


def test_unknown_launcher_is_not_misidentified(monkeypatch):
    root, _ = install_process_tree(monkeypatch, "unknown-launcher")

    match = utils._find_mpi_process(root.pid)

    assert match == {"type": None, "process": None}


def test_srun_uses_slurm_path(monkeypatch):
    root, _ = install_process_tree(monkeypatch, "srun")
    monkeypatch.setattr(utils, "_locate_slurm_step", lambda **kwargs: "12345.0")

    with pytest.warns(UserWarning, match="srun"):
        match = utils._find_mpi_process(root.pid)

    assert match == {"type": "slurm", "process": "12345.0"}


def test_mpi_controller_dispatches_signal(monkeypatch):
    root, launcher_process = install_process_tree(monkeypatch, "prte")

    controller = process_control.get_process_controller(
        root.pid, command="mpirun -np 2 vasp_std"
    )

    assert controller.kind == "mpi"
    assert controller.send_signal(19) is True
    assert launcher_process.signals == [19]


def test_srun_controller_does_not_require_psutil(monkeypatch):
    monkeypatch.setattr(utils, "psutil", None)
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setattr(utils, "_locate_slurm_step", lambda **kwargs: "12345.0")
    sent = []
    monkeypatch.setattr(
        utils, "_slurm_signal", lambda step_id, sig: sent.append((step_id, sig))
    )

    controller = process_control.get_process_controller(999, command="srun vasp_std")

    assert controller.kind == "slurm"
    assert controller.send_signal(19) is True
    assert sent == [("12345.0", 19)]


def test_mpi_controller_is_unavailable_without_psutil(monkeypatch):
    monkeypatch.setattr(utils, "psutil", None)

    with pytest.warns(UserWarning, match="psutil"):
        match = utils._find_mpi_process(999)
    controller = process_control.get_process_controller(
        999, command="mpirun -np 2 vasp_std"
    )

    assert match == {"type": None, "process": None}
    assert controller is None


def test_missing_psutil_does_not_claim_to_pause(monkeypatch):
    monkeypatch.setattr(utils, "psutil", None)
    calc = VaspInteractive(command="mpirun -np 2 vasp_std")
    calc.process = FakeProcess("shell")

    with pytest.warns(UserWarning):
        calc._pause_calc()

    assert calc.mpi_state is None
    calc.process = None
