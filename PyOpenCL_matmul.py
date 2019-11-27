# -*- coding: utf-8 -*-
#!/usr/bin/env python

#################################
# author = Drew Afromsky        #
# email = daa2162@columbia.edu  #
#################################

import numpy as np
import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pyopencl as cl
import pyopencl.array
import matplotlib.image as mpimg

class MatrixMultiply:

    def __init__(self):
        
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
            if platform.name == NAME:
                devs = platform.get_devices()
        
        # Set up a command queue:
        self.ctx = cl.Context(devs)
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # Set up class variables
        self.m1 = np.array(a_cpu, np.float32)
        self.m2 = np.array(b_cpu, np.float32)
        self.c = np.zeros((self.m1.shape[0],self.m2.shape[1]), dtype=np.float32)
       
        # Write the kernel code
        self.kernel_code_optimized = """
        __kernel void optimized(__global float*M, __global float*N, __global float*P, const int Widthx, const int Widthy)

        {
            __local float Mds[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];
            __local float Nds[%(BLOCK_SIZE)s][%(BLOCK_SIZE)s];

            int tx=get_local_id(0); 
            int ty=get_local_id(1); 
            int bx=get_group_id(0);
            int by=get_group_id(1);
            
            int Row = ty + by * %(BLOCK_SIZE)s;
            int Col = tx + bx * %(BLOCK_SIZE)s;
            
            float Pvalue = 0;

            for(int ph = 0; ph <= Widthy / %(BLOCK_SIZE)s; ph++) {
                if (Row < Widthx && ph * %(BLOCK_SIZE)s + tx < Widthy){
                    Mds[ty][tx]=M[Row * Widthy + ph * %(BLOCK_SIZE)s + tx];
                }
                else{
                    Mds[ty][tx]=0;    
                }
                if (Col < Widthx && ph * %(BLOCK_SIZE)s + ty < Widthy){
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
        __kernel void naive(__global float *M, __global float *N, __global float *P, const int Widthx, const int Widthy) {
            int index_x = get_global_id(0); 
            int index_y = get_global_id(1); 
            
            P[index_y * Widthx + index_x] = 0;
            if (index_x < Widthx) {
                for (int k=0; k < Widthy; k++) {
                    P[index_y * Widthx + index_x] += M[index_y * Widthy + k] * N[k * Widthx + index_x];
                }
            }
        }
        """
        self.kernel_code_optimized = self.kernel_code_optimized % {
        'BLOCK_SIZE':BLOCK_SIZE,
        }
        self.prg_naive = cl.Program(self.ctx, self.kernel_code_naive).build()
        self.prg_optimized = cl.Program(self.ctx, self.kernel_code_optimized).build()

    def matrix_mul_naive(self, a_cpu, b_cpu):

        m_cpu=max(self.m1.shape)            
        
        # Move variables to device       
        self.c_d = cl.array.to_device(self.queue, self.c)
        self.m1_d = cl.array.to_device(self.queue, self.m1)
        self.m2_d = cl.array.to_device(self.queue, self.m2)

        # Call kernel
        func_naive = self.prg_naive.naive

        # Measure time    
        # start = time.time()
        evt = func_naive(self.queue, (np.int32(m_cpu),np.int32(m_cpu)), None, self.m1_d.data, self.m2_d.data, self.c_d.data, np.int32(self.m2.shape[1]), np.int32(self.m1.shape[1]))
        evt.wait()
        # end = time.time()
        time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time
        
        # Return multiplied_matrix, kernel_execution_time
        multiplied_matrix = self.c_d.get()
        kernel_execution_time = time_

        return multiplied_matrix, kernel_execution_time
    
    def matrix_mul_optimized(self, a_cpu, b_cpu):
        m_cpu=max(self.m1.shape)   

        # Move variables to device       
        self.c_d = cl.array.to_device(self.queue, self.c)
        self.m1_d = cl.array.to_device(self.queue, self.m1)
        self.m2_d = cl.array.to_device(self.queue, self.m2)

        # Call kernel
        func_optimized = self.prg_optimized.optimized

        # Measure time    
        evt = func_optimized(self.queue,(np.int32(np.ceil(((m_cpu-1)|31)+1)),np.int32(np.ceil(((m_cpu-1)|31)+1))), (32,32), self.m1_d.data, self.m2_d.data, self.c_d.data, np.int32(self.m2.shape[1]), np.int32(self.m1.shape[1]))
        evt.wait()
        time_ = 1e-9 * (evt.profile.end - evt.profile.start) #this is the recommended way to record OpenCL running time
        
        # Return multiplied_matrix, kernel_execution_time
        multiplied_matrix = self.c_d.get()
        kernel_execution_time = time_

        return multiplied_matrix, kernel_execution_time

if __name__ == '__main__':

    cl_times_n = []
    cl_times_o = []
    py_times=[]
    n = np.arange(0,39,1)
    empty_array=[]
    # Create data 
    for itr in range(1,39):
        a_cpu = np.float32(np.random.randint(low=0, high=5, size=(itr*16,itr*14)))
        b_cpu = np.float32(np.random.randint(low=0, high=5, size=(itr*14,itr*16)))
        s1 = a_cpu.shape[0]
        # print("S1:", s1)
        # print("EMPTY ARRAY:", empty_array)
        empty_array.append(s1*s1)
        BLOCK_SIZE = 32

    # Create the output array
        cl_output_n = None # OPENCL
        cl_output_o= None
    # Create instance for OPENCL
        module = MatrixMultiply()

    # Serial (Python)
        s_times=[]
        for u in range(3):
            start=time.time()
            mat_mul=np.dot(a_cpu,b_cpu)
            s_times.append(time.time()-start)
        py_times.append(np.average(s_times))
    # OPENCL (naive)
        n_times=[]
        for e in range(3):
            cl_output_n, t = module.matrix_mul_naive(a_cpu,b_cpu)
            n_times.append(t)
        cl_times_n.append(np.average(n_times))

    # OPENCL (optimized)
        o_times=[]
        for q in range(3):
            cl_output_o, t1 = module.matrix_mul_optimized(a_cpu,b_cpu)
            o_times.append(t1)
        cl_times_o.append(np.average(o_times))
        
        # print("OPENCL OUTPUT NAIVE:", cl_output_n)
        # print("OPENCL OUTPUT OPTIMIZED:", cl_output_o)
        print("OPENCL NAIVE SHAPE:", cl_output_n.shape)
        print("OPENCL OPTIMIZED SHAPE:", cl_output_o.shape)
        # print('SERIAL OUTPUT:', np.dot(a_cpu,b_cpu))
        print("Code equality for CPU and naive GPU computation:", np.allclose(np.dot(a_cpu,b_cpu), cl_output_n))
        print("Code equality for CPU and naive GPU computation:", np.allclose(np.dot(a_cpu,b_cpu), cl_output_o))
        # print("OPENCL Times NAIVE:", cl_times_n)
        # print("OPENCL Times OPTIMIZED:", cl_times_o)
        # print("Serial Times:", py_times)
        # for w in range(len(py_times)):
        #     print("Speed-Up OPENCL (NAIVE):", py_times[w]/cl_times_n[w])
        #     print("Speed-Up OPENCL (OPTIMIZED):", py_times[w]/cl_times_o[w])
        
    MAKE_PLOT = True
    if MAKE_PLOT:
        plt.gcf()
        plt.plot(empty_array, py_times,'r', label="Python")
        plt.plot(empty_array, cl_times_o,'b', label="OPENCL OPTIMIZED")
        plt.plot(empty_array, cl_times_n,'g', label="OPENCL NAIVE")
        plt.legend(loc='upper left')
        plt.title('Matrix Multiplication')
        plt.xlabel('size of array')
        plt.ylabel('output coding times(sec)')
        plt.gca().set_xlim((min(empty_array), max(empty_array)))
        plt.savefig('plots_opencl.png')



