export PATH=/home/jzhao/cmake-3.18.2/install/bin/:${PATH}
module use /soft/modulefiles/
#module load public_intel_level_zero/release/master-2020.10.16
#module load intel_compute_runtime/release/master-2021.01.04
module load intel_compute_runtime
module load hiplz

export LD_LIBRARY_PATH=/home/jzhao/hipclworkspace/hipcl/install/lib/:${LD_LIBRARY_PATH}

module load iprof
module load intel_compute_runtime


