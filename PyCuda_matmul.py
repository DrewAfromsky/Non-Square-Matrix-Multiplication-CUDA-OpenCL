# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

from __future__ import division
import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
from pycuda import gpuarray, compiler, tools, cumath

import numpy as np
from numpy import linalg as la
 
class MatrixMultiply:

    def __init__(self):

        # Set up class variables
        self.m1 = np.array(a_cpu, np.float32)
        self.m2 = np.array(b_cpu, np.float32)

        # Write the kernel code
        self.kernel_code_optimized = """
        __global__ void optimized(float*M, float*N, float*P, const int Widthx, const int Widthy)

        {
            __shared__ float Mds[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
            __shared__ float Nds[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

            int tx=threadIdx.x; 
            int ty=threadIdx.y; 
            int bx=blockIdx.x;
            int by=blockIdx.y;
            
            int Row = ty + by * %(BLOCK_SIZE)s;
            int Col = tx + bx * %(BLOCK_SIZE)s;
            
            float Pvalue=0;

            for(int ph=0; ph<= Widthy/%(BLOCK_SIZE)s; ph++) {
                if (Row < Widthx && ph*%(BLOCK_SIZE)s + tx < Widthy){
                    Mds[ty][tx]=M[Row * Widthy + ph * %(BLOCK_SIZE)s + tx];
                }
                else{
                    Mds[ty][tx]=0;    
                }
                if (Col < Widthx && ph*%(BLOCK_SIZE)s + ty < Widthy){
                    Nds[ty][tx]=N[(ph * %(BLOCK_SIZE)s + ty)* Widthx + Col];
                }
                else{
                    Nds[ty][tx]=0;
                }
                __syncthreads();
                
                for (int k = 0; k < %(BLOCK_SIZE)s; ++k) {
                    Pvalue += Mds[ty][k] * Nds[k][tx];
                }
                __syncthreads();
                }
            if (Row < Widthx && Col < Widthx) {
                P[Row * Widthx + Col] = Pvalue;
            }
        } """

        self.kernel_code_naive = """
        __global__ void naive(float *M, float *N, float *P, const int Widthx, const int Widthy) {
            int Row = blockIdx.y * blockDim.y + threadIdx.y;
            int Col = blockIdx.x * blockDim.x + threadIdx.x;
            
            P[Row * Widthx + Col] = 0;
            if (Col < Widthx) {
                for (int k=0; k < Widthy; k++) {
                    P[Row * Widthx + Col] += M[Row * Widthy + k] * N[k * Widthx + Col];
                }
            }
        }
        """

        self.kernel_code = self.kernel_code_optimized % {
            'BLOCK_SIZE':BLOCK_SIZE,
            }
        self.mod_o = compiler.SourceModule(self.kernel_code)

        self.mod_n = compiler.SourceModule(self.kernel_code_naive)
        # self.mod_o = compiler.SourceModule(self.kernel_code_optimized)

    def matrix_mul_naive(self, a_cpu, b_cpu):
        # Move variables to device
        self.m1_d = gpuarray.to_gpu(self.m1)
        self.m2_d = gpuarray.to_gpu(self.m2)
        self.c_d = gpuarray.empty((self.m1.shape[0],self.m2.shape[1]), np.float32)
        
        # create CUDA Event to measure time
        start = cuda.Event()
        end = cuda.Event()

        # Call kernel
        func_n = self.mod_n.get_function('naive')
        start.record()
        start_=time.time()
        # Measure time
        grid_dim_x = np.ceil(np.float32(max(self.m2_d.shape[0], self.m1_d.shape[0]))/32)
        func_n(self.m1_d, self.m2_d, self.c_d, np.int32(self.m2.shape[1]), np.int32(self.m1.shape[1]), block=(32, 32, 1), grid = (np.int(grid_dim_x),np.int(grid_dim_x),1))
        end_ = time.time()
        end.record()
        
        # memory copy to host
        # cuda.memcpy_dtoh(self.c, self.c_d)

        # CUDA Event synchronize
        end.synchronize()

        multiplied_matrix = self.c_d.get()
        kernel_execution_time = end_-start_

        # Return multiplied_matrix, kernel_execution_time
        return multiplied_matrix, kernel_execution_time
        # Measure time    
        # Return multiplied_matrix, kernel_execution_time

    def matrix_mul_optimized(self, a_cpu, b_cpu):

        # Move variables to device
        # self.m1_d = cuda.mem_alloc(self.m1.nbytes)
        # self.m2_d = cuda.mem_alloc(self.m2.nbytes)
        # self.c_d = cuda.mem_alloc(self.c.nbytes)
        self.m1_d = gpuarray.to_gpu(self.m1)
        self.m2_d = gpuarray.to_gpu(self.m2)
        self.c_d = gpuarray.empty((self.m1.shape[0],self.m2.shape[1]), np.float32)
        
        # copy data to device
        # cuda.memcpy_htod(self.m1_d, self.m1)
        # cuda.memcpy_htod(self.m2_d, self.m2)
        # cuda.memcpy_htod(self.c_d, self.c)

        # create CUDA Event to measure time
        start = cuda.Event()
        end = cuda.Event()

        # Call kernel
        func_o = self.mod_o.get_function('optimized')
        start.record()
        start_=time.time()
        # Measure time
        grid_dim_x = np.ceil(np.float32(max(self.m2_d.shape[0], self.m1_d.shape[0]))/32)
        func_o(self.m1_d, self.m2_d, self.c_d, np.int32(self.m2.shape[1]), np.int32(self.m1.shape[1]), block=(32, 32, 1), grid = (np.int(grid_dim_x),np.int(grid_dim_x),1))
        end_ = time.time()
        end.record()
        
        # memory copy to host
        # cuda.memcpy_dtoh(self.c, self.c_d)

        # CUDA Event synchronize
        end.synchronize()

        multiplied_matrix = self.c_d.get()
        kernel_execution_time = end_-start_
        # Return multiplied_matrix, kernel_execution_time
        return multiplied_matrix, kernel_execution_time

if __name__ == '__main__':

    cu_times_n = []
    cu_times_o = []
    py_times=[]
    n = np.arange(0,39,1)
    empty_array=[]
    # Create data 
    for itr in range(1,39):
        a_cpu = np.float32(np.random.randint(low=0, high=5, size=(itr*16,itr*14)))
        b_cpu = np.float32(np.random.randint(low=0, high=5, size=(itr*14,itr*16)))
        s1 = a_cpu.shape[0]
        empty_array.append(s1*s1)
        # TILE_SIZE = 2
        BLOCK_SIZE = 32

    # Create the output array
        cu_output_n = None # CUDA
        cu_output_o= None
    # Create instance for CUDA
        module = MatrixMultiply()

    # Serial (Python)
        s_times=[]
        for u in range(3):
            start=time.time()
            mat_mul=np.dot(a_cpu,b_cpu)
            s_times.append(time.time()-start)
        py_times.append(np.average(s_times))
    # CUDA (naive)
        n_times=[]
        for e in range(3):
            cu_output_n, t = module.matrix_mul_naive(a_cpu,b_cpu)
            n_times.append(t)
        cu_times_n.append(np.average(n_times))

    # CUDA (optimized)
        o_times=[]
        for q in range(3):
            cu_output_o, t1 = module.matrix_mul_optimized(a_cpu,b_cpu)
            o_times.append(t1)
        cu_times_o.append(np.average(o_times))
        
        print("CUDA NAIVE SHAPE:", cu_output_n.shape)
        print("CUDA OPTIMIZED SHAPE:", cu_output_o.shape)
        print("Code equality for CPU and naive GPU computation:", np.allclose(np.dot(a_cpu,b_cpu), cu_output_n))
        print("Code equality for CPU and optimized GPU computation:", np.allclose(np.dot(a_cpu,b_cpu), cu_output_o))
        # print("EMPTY ARRAY:", empty_array)
        # print("CUDA Times NAIVE:", cu_times_n)
        # print("CUDA Times OPTIMIZED:", cu_times_o)
        # print("Serial Times:", py_times)
        # for w in range(len(py_times)):
        #     print("Speed-Up CUDA (NAIVE):", py_times[w]/cu_times_n[w])
        #     print("Speed-Up CUDA (OPTIMIZED):", py_times[w]/cu_times_o[w])
        
    MAKE_PLOT = True
    if MAKE_PLOT:
        plt.gcf()
        plt.plot(empty_array, py_times,'r', label="Python")
        plt.plot(empty_array, cu_times_o,'b', label="CUDA OPTIMIZED")
        plt.plot(empty_array, cu_times_n,'g', label="CUDA NAIVE")
        plt.legend(loc='upper right')
        plt.title('Matrix Multiplication')
        plt.xlabel('size of array')
        plt.ylabel('output coding times(sec)')
        plt.gca().set_xlim((min(empty_array), max(empty_array)))
        plt.savefig('plots_pycuda.png')
