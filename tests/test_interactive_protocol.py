import io

import pytest
from ase import Atoms

from vasp_interactive import VaspInteractive


class _MockStdout(io.StringIO):
    def feed(self, text):
        position = self.tell()
        self.seek(0, io.SEEK_END)
        super().write(text)
        self.seek(position)


class _MockStdin(io.StringIO):
    def __init__(self, process):
        super().__init__()
        self.process = process

    def write(self, text):
        count = super().write(text)
        for line in text.splitlines():
            self.process.accept_line(line)
        return count


class MockVaspProcess:
    """Small protocol-only VASP stand-in; it never runs an external process."""

    def __init__(self, version, counts=(2, 1)):
        self.version = version
        self.counts = tuple(counts)
        self.n_atoms = sum(counts)
        self.protocol = self._protocol_for_version(version)
        self.stdout = _MockStdout(
            f" vasp.{version} mock-build\n"
            + self._reading_prompt
        )
        self.stdin = _MockStdin(self)
        self.received = []
        self.accepted = False

    @staticmethod
    def _protocol_for_version(version):
        parts = tuple(int(part) for part in version.split(".")[:3])
        return "poscar" if parts >= (6, 4, 1) else "positions"

    @property
    def _reading_prompt(self):
        if self.protocol == "poscar":
            return "POSITIONS AND LATTICE: reading from stdin\n"
        return "POSITIONS: reading from stdin\n"

    def poll(self):
        return None

    def accept_line(self, line):
        if self.accepted:
            raise AssertionError("mock VASP received input after the record ended")
        self.received.append(line)
        if self.protocol == "positions":
            self._accept_positions_record()
        else:
            self._accept_poscar_record()

    def _accept_positions_record(self):
        if len(self.received) > self.n_atoms:
            raise AssertionError("legacy VASP accepts positions only")
        try:
            values = [float(value) for value in self.received[-1].split()]
        except ValueError as error:
            raise AssertionError("legacy VASP received a non-numeric line") from error
        if len(values) != 3:
            raise AssertionError("legacy VASP expects three position values per line")
        if len(self.received) == self.n_atoms:
            self.accepted = True
            self.stdout.feed(
                "POSITIONS: read from stdin\n"
                "next electronic-step line\n"
            )

    def _accept_poscar_record(self):
        index = len(self.received) - 1
        if index == 0:
            if not self.received[-1]:
                raise AssertionError("POSCAR title is empty")
        elif index == 1:
            if len(self.received[-1].split()) != 1:
                raise AssertionError("POSCAR scale line is invalid")
        elif 2 <= index <= 4:
            if len(self.received[-1].split()) != 3:
                raise AssertionError("POSCAR lattice line is invalid")
        elif index == 5:
            if not self.received[-1].strip():
                raise AssertionError("POSCAR species line is empty")
        elif index == 6:
            if tuple(int(value) for value in self.received[-1].split()) != self.counts:
                raise AssertionError("POSCAR atom counts are invalid")
        elif index == 7:
            if self.received[-1].lower() != "direct":
                raise AssertionError("POSCAR coordinate mode is invalid")
        elif 8 <= index < 8 + self.n_atoms:
            if len(self.received[-1].split()) != 3:
                raise AssertionError("POSCAR position line is invalid")
        else:
            raise AssertionError("POSCAR record contains too many lines")

        if len(self.received) == 8 + self.n_atoms:
            self.accepted = True
            self.stdout.feed(
                "POSITIONS AND LATTICE: read from stdin\n"
                "next electronic-step line\n"
            )


def _atoms():
    return Atoms(
        "OH2",
        positions=[(0, 0, 0), (0.7, 0.7, 0), (0.7, 0, 0.7)],
        cell=[8, 8, 8],
        pbc=True,
    )


def _calculator(process, directory):
    calc = VaspInteractive(
        directory=directory,
        txt=None,
        allow_mpi_pause=False,
    )
    calc.process = process
    # ASE's normal initialization creates this ordering and count list.
    calc.sort = [1, 2, 0]
    calc.symbol_count = [("H", 2), ("O", 1)]
    return calc


def _start_mock_calculation(calc, atoms):
    calc._run(atoms, out=None, require_cell_stdin=False)
    assert calc._interactive_protocol == calc.process.protocol


def test_pre_64_vasp_accepts_positions_only(tmp_path):
    process = MockVaspProcess("6.4.0")
    calc = _calculator(process, tmp_path)
    try:
        atoms = _atoms()
        _start_mock_calculation(calc, atoms)
        calc._write_atoms_stdin(atoms, out=None, require_cell_stdin=False)
        assert process.protocol == "positions"
        assert process.accepted
        assert len(process.received) == 3
        assert calc._pending_stdout == "next electronic-step line\n"
    finally:
        calc.process = None


@pytest.mark.parametrize("version", ["6.4.1", "6.5.1", "6.6.0"])
def test_64_plus_vasp_accepts_full_poscar(tmp_path, version):
    process = MockVaspProcess(version)
    calc = _calculator(process, tmp_path)
    try:
        atoms = _atoms()
        _start_mock_calculation(calc, atoms)
        calc._write_atoms_stdin(atoms, out=None, require_cell_stdin=True)
        assert process.protocol == "poscar"
        assert process.accepted
        assert process.received[0].strip() == "H  O"
        assert process.received[7] == "Direct"
        assert len(process.received) == 11
    finally:
        calc.process = None
