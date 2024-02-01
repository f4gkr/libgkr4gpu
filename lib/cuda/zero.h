/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank - lib gkr4gpu                       *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#ifndef ZERO_H
#define ZERO_H

#include <cuda_runtime.h>
#include "datatypes.h"

// On some old generation Jetson like TX1 and Jetson NANO the __shfl_xor_sync intrisics do not exist
// you have to define CUDA_OLD to have the code compile

#ifndef GPU_BLOCK_SIZE
#ifdef __aarch64__
#define GPU_BLOCK_SIZE (256)
#else
#define GPU_BLOCK_SIZE (256)
#endif
#endif

struct GPUChannelizer
{
    float mix_offset;
    float mix_phase;
    int DecimFactor;
    int oversampling;
    long long decim_r_pos;
    long overlap;
    int32_t fft_size;
    int32_t decimated_sample_count;
};

void zero(float *l_p_array, int a_numElements);
void GPUconvolve(CF32 *a, CF32 *b, CF32 *destination,
                 int size, cudaStream_t streamID);
void GPUpowerOfComplexVector(CF32 *in, float *out, int size, cudaStream_t streamID);

void generateOscillator(CF32 *dest, float startphase, float increment, int L, int gap,
                        cudaStream_t streamID);
void gpuDecimate(CF32 *in, CF32 *out, int *p, int L, cudaStream_t streamID);

void multiChannel(GPUChannelizer *channel, int L,
                  CF32 *input, CF32 *output,
                  char *output_valid,
                  cudaStream_t streamID); 
#endif // ZERO_H
