#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

# Ensure we don't have the NVVM and NVRTC libraries available
apt-get update
apt remove --purge `dpkg --get-selections | grep cuda-nvvm | awk '{print $1}'` -y
apt remove --purge `dpkg --get-selections | grep cuda-nvrtc | awk '{print $1}'` -y

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n test \
    psutil \
    pytest \
    cffi \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-mamba-retry install -c `pwd`/conda-repo numba-cuda

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"
pushd "${RAPIDS_TESTS_DIR}"

rapids-print-env

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Run Tests"
export NUMBA_ENABLE_CUDASIM=1
python -m numba.runtests numba.cuda.tests -v

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
