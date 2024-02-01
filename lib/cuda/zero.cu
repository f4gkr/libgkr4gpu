/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h> // <-- added for 'printf'
#include <math.h>
#include "zero.h"

typedef struct
{
    double I;
    double Q;
} GCF32;

__global__ void zero_GPU(float *l_p_array_gpu)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // <-- in case you use more blocks
    l_p_array_gpu[i] = 0.;
}


void zero(float *l_p_array, int a_numElements)
{
    float *l_p_array_gpu;

    int size = a_numElements * int(sizeof(float));

    cudaMalloc((void**) &l_p_array_gpu, size);
    cudaMemcpy(l_p_array_gpu, l_p_array, size, cudaMemcpyHostToDevice);

    // use one block with a_numElements threads
    zero_GPU<<<1, a_numElements>>>(l_p_array_gpu);

    cudaMemcpy(l_p_array, l_p_array_gpu, size, cudaMemcpyDeviceToHost);
    cudaFree(l_p_array_gpu);
}

__global__ void KgenerateOscillator( CF32 *dest, float startphase, float increment, int L, int gap ) {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if( i < L ) {
         if( i >= gap ) {
            float angle = startphase + i*increment ;
            dest[i].I = cosf(angle);
            dest[i].Q = sinf(angle);
         }
     }
}

void generateOscillator( CF32 *dest, float startphase, float increment,
                         int L, int gap, cudaStream_t streamID ) {
    int block_size, grid_size ;
    block_size = GPU_BLOCK_SIZE ;
    grid_size = (int)ceilf((float)L/block_size);
    KgenerateOscillator<<<grid_size, block_size, 0, streamID>>>(dest,startphase, increment, L, gap);
}



__global__ void kDecimate(CF32* __restrict__ signal,
                          CF32* __restrict__ out,
                          int* __restrict__ positions,
                          int L ) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < L ) {
        if( positions[i] >= 0 )
            out[i] = signal[ positions[i] ];
    }
}

void gpuDecimate( CF32* in, CF32* out, int *p, int L, cudaStream_t streamID ) {
    int block_size, grid_size ;
    block_size = GPU_BLOCK_SIZE ;
    grid_size = (int)ceilf((float)L/block_size);
    kDecimate<<<grid_size, block_size, 0, streamID>>>(in,out,p,L);
}

////////////////////////////////////////////////////////////////////////////////
// Complex operations
////////////////////////////////////////////////////////////////////////////////

// Complex addition
static __device__ __host__ inline CF32 ComplexAdd(CF32 a, CF32 b)
{
    CF32 c;
    c.I = a.I + b.I;
    c.Q = a.Q + b.Q;
    return c;
}

// Complex scale
static __device__ __host__ inline CF32 ComplexScale(CF32 a, float s)
{
    CF32 c;
    c.I = s * a.I;
    c.Q = s * a.Q;
    return c;
}

// Complex multiplication
static __device__ __host__ inline CF32 ComplexMul(CF32 a, CF32 b)
{
    CF32 c;
    c.I = a.I * b.I - a.Q * b.Q;
    c.Q = a.I * b.Q + a.Q * b.I;
    return c;
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMul( CF32* __restrict__ a, CF32* __restrict__ b,
                                           CF32* __restrict__ destination, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    CF32 _a, _b ;
    if( i < size ) {
        _a = a[i] ;
        _b = b[i] ;
        destination[i].I = _a.I * _b.I - _a.Q * _b.Q;
        destination[i].Q = _a.I * _b.Q + _a.Q * _b.I;
    }
}

void GPUconvolve(CF32 *a, CF32 *b, CF32* destination,
                 int size, cudaStream_t streamID) {
    int block_size, grid_size ;
    block_size = GPU_BLOCK_SIZE ;
    grid_size = (int)ceilf((float)size/block_size);
    ComplexPointwiseMul<<<grid_size, block_size, 0, streamID>>>(a, b, destination, size );
}


// WARNING  !!!!!!!!!!!!
// do not replace double precision by single precision unless you know what you are doing...
// the GPU sin() and cos() functions are VERY noisy on some platforms when in single precision
// TEST !!!

static __global__ void multiChannelKernel( GPUChannelizer * channel,
                                    CF32*   input,
                                    CF32*  output,
                                    char*  output_valid ) {
     int n,m;
     int index = blockIdx.x * blockDim.x + threadIdx.x;
     int L = channel->decimated_sample_count  ;
     if( index >= L )
         return ;

     GCF32 b ;
     GCF32 in ;

     int p = channel->decim_r_pos % (channel->fft_size - channel->overlap)  ;
     p += (index * channel->DecimFactor ) / channel->oversampling ;
     int q = p + channel->overlap ;

     double factor = 1.0 / channel->fft_size ;

    if( q < channel->fft_size ) {

         double tmp_phase = channel->mix_phase + p*channel->mix_offset ;

         n = floorf( fabs( tmp_phase )/(2*M_PI)) ;
         if( n ) {
             m = (tmp_phase > 0) ? -n:n;
             tmp_phase += m * 2 * M_PI ;
         }

         b.I = cos(tmp_phase) * factor;
         b.Q = sin(tmp_phase) * factor;

         in.I = (double)input[q].I ;
         in.Q = (double)input[q].Q ;

         output[index].I = (float)(in.I * b.I - in.Q * b.Q);
         output[index].Q = (float)(in.I * b.Q + in.Q * b.I);

         output_valid[index] = (char)1 ;
     }

}

void multiChannel(GPUChannelizer * channel, int L,
                  CF32*   input, CF32 *  output, char*  output_valid,
                  cudaStream_t streamID ) {

    int block_size, grid_size ;
    block_size = GPU_BLOCK_SIZE ;
    grid_size = (int)ceilf((float)L/block_size) ;
    multiChannelKernel<<<grid_size, block_size, 0, streamID>>>(channel,input,output,output_valid);
}


__global__ void powerOfComplex( CF32 *cpxin, float *output, int numOfCpx ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    CF32 b ;
    float module ;
    if( index < numOfCpx ) {
        b = cpxin[index] ;
        module = ( b.I*b.I + b.Q*b.Q);
        output[index] = module ;
    }
}

void GPUpowerOfComplexVector( CF32 *in, float *out, int size, cudaStream_t streamID) {
    int block_size, grid_size ;
    block_size = GPU_BLOCK_SIZE ;
    grid_size = (int)ceilf((float)size/block_size) ;
    powerOfComplex<<<grid_size, block_size, 0, streamID>>>(in, out, size);
}

__global__ void LOGpowerOfComplex( CF32 *cpxin, float *output, int numOfCpx ) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    CF32 b ;
    float module ;
    if( index < numOfCpx ) {
        b = cpxin[index] ;
        module = 20*log10f( 1.0/numOfCpx * (b.I*b.I + b.Q*b.Q));
        output[index] = module ;
    }
}

void GPULOGpowerOfComplexVector( CF32 *in, float *out, int size, cudaStream_t streamID) {
    int block_size, grid_size ;
    block_size = GPU_BLOCK_SIZE ;
    grid_size = (int)ceilf((float)size/block_size) ;
    LOGpowerOfComplex<<<grid_size, block_size, 0, streamID>>>(in, out, size);
}



#define WARP_SIZE 32 // # of threads that are executed together (constant valid on most hardware)
__global__ void sumOfFloats(float * in, float * output, int num_elements) {
    //Holds intermediates in shared memory reduction
        __syncthreads();
        __shared__ float buffer[WARP_SIZE];
        int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
        int lane = threadIdx.x % WARP_SIZE;
        float temp;

        while(globalIdx < num_elements)
        {
            // All threads in a block of 1024 take an element
            temp = in[globalIdx];
            // All warps in this block (32) compute the sum of all
            // threads in their warp
            for(int delta = WARP_SIZE/2; delta > 0; delta /= 2)
            {
#ifdef CUDA_OLD
                 temp+= __shfl_xor(temp, delta);
#else
                 temp+= __shfl_xor_sync(0xFFFFFFFF, temp, delta);
#endif
            }
            // Write all 32 of these partial sums to shared memory
            if(lane == 0)
            {
                buffer[threadIdx.x / WARP_SIZE] = temp;
            }
            __syncthreads();
            // Add the remaining 32 partial sums using a single warp
            if(threadIdx.x < WARP_SIZE)
            {
                temp = buffer[threadIdx.x];
                for(int delta = WARP_SIZE / 2; delta > 0; delta /= 2)
                {
#ifdef CUDA_OLD
                 temp+= __shfl_xor(temp, delta);
#else
                 temp+= __shfl_xor_sync(0xFFFFFFFF, temp, delta);
#endif
                }
            }
            // Add this block's sum to the total sum
            if(threadIdx.x == 0)
            {
                atomicAdd(output, temp);
            }
            // Jump ahead 1024 * #SMs to the next region of numbers to sum
            globalIdx += blockDim.x * gridDim.x;
            __syncthreads();
    }
}

void GPUSumOfFloats( float *in, float *out, int numInputElements, int num_SMs, cudaStream_t streamID) {
    sumOfFloats<<<num_SMs, 512, 0, streamID>>>(in, out, numInputElements);
}

int GPUSumOfFloatsPadding(int numInputElements, int *num_SMs ) {
    // Get device properties to compute optimal launch bounds
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    *num_SMs = prop.multiProcessorCount;
    int batch_size = (*num_SMs) * 512;
    int padding = (batch_size - (numInputElements % batch_size)) % batch_size;
    return( padding );
}

 