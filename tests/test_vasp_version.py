import pytest
from vasp_interactive import VaspInteractive
from ase.calculators.calculator import CalculatorSetupError
import tempfile
from pathlib import Path
import os
from ase.atoms import Atoms

d = 0.9575
h2_root = Atoms("H2", positions=[(d, 0, 0), (0, 0, 0)], cell=[8, 8, 8], pbc=True)
rootdir = Path(__file__).parents[1] / "sandbox"
fmax = 0.05
ediff = 1e-4


# Since version 0.0.8 using VaspInteractive on VASP 5.x will only raise Warning instead of Exception
@pytest.mark.filterwarnings("error:Some builds")
def test_steps():
    from ase.build import molecule

    """Test if VaspInteractive correctly write inputs
    """
    h2 = h2_root.copy()
    with tempfile.TemporaryDirectory() as tempdir:
        tempdir = Path(tempdir)
        calc = VaspInteractive(xc="pbe", ediff=ediff, directory=tempdir)
        h2.calc = calc
        with calc:
            # Initialization
            try:
                h2.get_potential_energy()
            except UserWarning:
                assert calc.version[0] == "5"
    return


# The lines below are taken verbatim from a real VASP 6.6.0 OUTCAR. The banner
# is printed whenever NCORE exceeds the number of cores per socket, so whether a
# run hits it depends on the parallel settings, not on the VASP version.
VERSION_LINE = " vasp.6.6.0 06Mar2026 (build Aug 29 2026 13:20:05) complex"
NCORE_URL_LINE = (
    "|     own testing! More info at https://www.vasp.at/wiki/index.php/NCORE      |"
)


def _bare_calc(tempdir):
    """A calculator that has never launched VASP, for parsing-only tests."""
    return VaspInteractive(xc="pbe", ediff=ediff, directory=tempdir)


def test_version_parsed_from_banner():
    """The real version banner is parsed."""
    with tempfile.TemporaryDirectory() as tempdir:
        calc = _bare_calc(tempdir)
        assert calc._read_vasp_version_stream(VERSION_LINE) is True
        assert calc.version == "6.6.0"


def test_wiki_url_is_not_a_version():
    """A vasp.at documentation URL must not be read as a version.

    Regression test: the NCORE performance-advice banner contains
    https://www.vasp.at/wiki/index.php/NCORE. A pattern of ``vasp\\.([^\\s]+)``
    matches inside ``www.vasp.at`` and captures ``at/wiki/index.php/NCORE``,
    which then raises ValueError in ``_int_version``.
    """
    from vasp_interactive.utils import _int_version

    with tempfile.TemporaryDirectory() as tempdir:
        calc = _bare_calc(tempdir)
        assert calc._read_vasp_version_stream(NCORE_URL_LINE) is False
        assert calc.version is None

        # and the whole sequence, as it arrives from a real run
        calc = _bare_calc(tempdir)
        for line in (VERSION_LINE, NCORE_URL_LINE):
            if calc._read_vasp_version_stream(line):
                _int_version(calc.version)  # must not raise
        assert calc.version == "6.6.0"


def test_version_is_not_overwritten():
    """Only the first parse for a given process is kept."""
    with tempfile.TemporaryDirectory() as tempdir:
        calc = _bare_calc(tempdir)
        assert calc._read_vasp_version_stream(VERSION_LINE) is True
        assert calc._read_vasp_version_stream(" vasp.5.4.4pl2 other build") is False
        assert calc.version == "6.6.0"


def test_version_5_still_detected():
    """The VASP 5 warning path is unaffected."""
    from vasp_interactive.utils import _int_version

    with tempfile.TemporaryDirectory() as tempdir:
        calc = _bare_calc(tempdir)
        assert calc._read_vasp_version_stream(" vasp.5.4.4pl2 complex") is True
        assert calc.version == "5.4.4pl2"
        assert _int_version(calc.version) == 5
