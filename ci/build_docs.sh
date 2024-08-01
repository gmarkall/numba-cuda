#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Install testing dependencies"
# TODO: Replace with rapids-dependency-file-generator
rapids-mamba-retry create -n docs \
    make \
    psutil \
    sphinx \
    sphinx_rtd_theme \
    python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

package=$(realpath conda-package/numba-cuda-*.tar.bz2)
echo "Package path: $package"
rapids-mamba-retry install $package

rapids-print-env

rapids-logger "Show Numba system info"
python -m numba --sysinfo

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "Build docs"
pushd docs
make html

popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}