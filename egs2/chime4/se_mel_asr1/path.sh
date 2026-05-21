MAIN_ROOT=$PWD/../../..

export PATH=$PWD/utils/:$PATH
export LC_ALL=C

if [ -f "${MAIN_ROOT}"/tools/activate_python.sh ]; then
    . "${MAIN_ROOT}"/tools/activate_python.sh
else
    echo "[INFO] ${MAIN_ROOT}/tools/activate_python.sh is not present"
fi
. "${MAIN_ROOT}"/tools/extra_path.sh

export OMP_NUM_THREADS=1
export PYTHONIOENCODING=UTF-8
export NCCL_SOCKET_IFNAME="^lo,docker,virbr,vmnet,vboxnet"

. local/path.sh
