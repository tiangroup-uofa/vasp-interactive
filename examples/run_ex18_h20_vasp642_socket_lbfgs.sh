#!/usr/bin/env bash
# H20 launcher for ex18_vasp642_socket_lbfgs.py.
# Load the site's NVHPC/OpenMPI/MKL environment before invoking this script.
set -euo pipefail

: "${VASP_COMMAND:?Set VASP_COMMAND to the tested VASP 6.4.2 launch command.}"
: "${VASP_PP_PATH:?Set VASP_PP_PATH to an authorized pseudopotential root.}"
: "${VPI_EXAMPLE_WORKDIR:?Set VPI_EXAMPLE_WORKDIR to a writable output directory.}"

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-14}
export OMP_PLACES=${OMP_PLACES:-cores}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}

exec python "$(dirname "$0")/ex18_vasp642_socket_lbfgs.py"
