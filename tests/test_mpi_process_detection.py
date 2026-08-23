import pytest

from vasp_interactive import utils


class FakeProcess:
    _next_pid = 1000

    def __init__(self, name, children=()):
        self._name = name
        self._children = list(children)
        self.pid = FakeProcess._next_pid
        FakeProcess._next_pid += 1

    def name(self):
        return self._name

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
