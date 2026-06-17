from __future__ import annotations

# ASE 4 experimental plugin declaration for vasp-interactive.
# This module is loaded through the ``ase.plugins`` entry point and uses the
# ``ase._4.plugins`` plugin system introduced for ASE 4 calculator discovery.

try:
    from ase._4.plugins.calculator import CalculatorPlugin
except ImportError:
    # ASE < 4 does not provide the new plugin system. Older ASE versions will
    # not consume the ``ase.plugins`` entry point, but keeping this module
    # importable avoids surprising failures if users/tools import it directly.
    __ase_plugins__ = set()
else:
    vasp_interactive_plugin = CalculatorPlugin(
        name="vasp-interactive",
        long_name="Stream-based ASE calculator for VASP interactive mode (enhanced version from VASP internal implementation)",
        citation="VaspInteractive developers",
        implementation="vasp_interactive.vasp_interactive.VaspInteractive",
        configurable=True,  # uses ASE/VASP executable and environment config
    )

    __ase_plugins__ = {vasp_interactive_plugin}
