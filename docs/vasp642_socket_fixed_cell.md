# VASP 6.4.2 fixed-cell socket example

`examples/ex18_vasp642_socket_lbfgs.py` is a small ASE LBFGS acceptance test
for the `phorbol/vasp642-socket` branch.  It creates a deterministic rattled
eight-atom diamond-Si cell, runs VASP through `VaspInteractive` and
`SocketIOCalculator`, and records a self-contained output directory.

It is an example of the **fixed-cell** VASP interactive protocol only.  It is
not suitable for `UnitCellFilter`, external equation-of-state workflows, or
any ASE optimization that changes `atoms.cell`.

## Prerequisites

The user must provide a compatible VASP executable and licensed
pseudopotentials.  This repository must not contain a VASP binary, source code,
or POTCAR files.

Set these variables after loading the site compiler/MPI/GPU environment:

```bash
export VASP_COMMAND='mpirun -np 1 /path/to/vasp_std'
export VASP_PP_PATH=/path/to/authorized/vasp-pseudopotentials
export VPI_EXAMPLE_WORKDIR=$PWD/vasp642-socket-runs
```

For the H20 configuration validated by this branch, use one GPU and one MPI
rank, then set `OMP_NUM_THREADS=14`, `OMP_PLACES=cores`, and
`OMP_PROC_BIND=close`.  The repository launcher applies those values only when
they have not already been set:

```bash
bash examples/run_ex18_h20_vasp642_socket_lbfgs.sh
```

The Python example avoids `ase.build` and requires a normal ASE installation
that provides `Atoms`, `LBFGS`, and `SocketIOCalculator`.

## What the example verifies

1. A VASP process remains alive while ASE sends several coordinate sets over a
   Unix-domain i-PI socket.
2. VASP returns finite DFT energy and forces to ASE.
3. A three-step external LBFGS calculation reaches `fmax <= 0.10 eV/Ang` for
   the deterministic Si test.
4. VASP 6.4.2 receives fixed-cell coordinate input using the
   `POSITIONS: reading from stdin` protocol.

The known H20 validation used VASP 6.4.2 with one GPU, NVHPC OpenACC, CUDA 12.4,
and CUDA-aware OpenMPI 5.  Four fixed-cell structures were independently
recomputed without `INTERACTIVE`; the maximum absolute energy difference was
`3.42e-6 eV` and the maximum force-component difference was `1.19e-3 eV/Ang`
at `EDIFF=1e-5`.

## Important limitations

### Fixed cell only

VASP 6.4.2 requests a complete POSCAR over stdin when `ISIF >= 3`.  The current
wrapper deliberately enforces `ISIF=2` and sends fractional coordinates only.
Cell-relaxation requests are rejected rather than silently producing an
incorrect result.  Supporting variable cells requires a separate implementation
and numerical regression test for the complete-POSCAR protocol.

### Electronic-state reuse

The VASP process is resident across ionic configurations.  The validated setup
uses `IWAVPR=11`: this is simple **charge-density** extrapolation, not second-
order orbital plus charge extrapolation.  Do not change this to `IWAVPR=12`
without a system-specific accuracy and stability test.

### One shutdown SCF is expected

When the `SocketIOCalculator` context closes, `VaspInteractive` writes a
`STOPCAR` and sends the last positions so VASP can observe the stop request.
VASP may perform one additional, normally very short SCF cycle before exiting.
That cycle is not requested by ASE and its force is not used by the optimizer.
It is a graceful-shutdown protocol cost, not a cache miss or an additional ASE
optimization step.

## Output and cleanup

Each invocation creates a fresh `vasp642-socket-lbfgs-*` directory below
`VPI_EXAMPLE_WORKDIR`, containing `summary.json`, `lbfgs.log`, `lbfgs.traj`,
socket logs, and normal VASP output.  Do not publish POTCAR, WAVECAR, CHGCAR,
or licensed VASP artifacts from such a directory.
