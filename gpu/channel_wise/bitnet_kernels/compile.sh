#nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo --shared bitnet_kernels.cu -lcuda -gencode=arch=compute_86,code=sm_86 -o libbitnet.so
nvcc -std=c++17 -Xcudafe --diag_suppress=177 --compiler-options -fPIC -lineinfo -Xptxas --make-errors-visible-at-exit --shared bitnet_kernels.cu -lcuda -gencode=arch=compute_86,code=sm_86 -o libbitnet.so


