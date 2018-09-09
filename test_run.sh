#!/bin/bash
SM="50"
CODE="sm_$SM"
COMPUTE="compute_$SM"
PROFILE_FROM_START="off"

nvcc test_kernel.cu -o test_kernel -gencode arch=$COMPUTE,code=$CODE -gencode arch=$COMPUTE,code=$COMPUTE -lnvToolsExt
nvprof -f --profile-from-start $PROFILE_FROM_START -o test_kernel_profile.nvvp ./test_kernel

python correlate.py test_kernel_profile.nvvp
rm test_kernel test_kernel_profile.nvvp
