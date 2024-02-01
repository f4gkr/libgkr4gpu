/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#ifndef OLASGPUCHANNEL_H
#define OLASGPUCHANNEL_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <unistd.h>
#include <mutex>

#include "datatypes.h"
#include "cuda/zero.h" 

#define INPUT_TOO_LONG (-1)
#define NEED_MORE_DATA (-2)
#define NO_FEEDER (-3)
#define GET_DATA_OUT (1)
#define NO_OUT_DATA (-1)
#define NOT_ENOUGH_DATA (-1)

class DDCBankChannel
{
public:
    static int FILTER_KERNEL_SIZE ;

    DDCBankChannel(double inSampleRate,
                   double outSampleRate,
                   int oversampling,
                   int maxInSize,
                   unsigned int use_fft_size );
    ~DDCBankChannel();

    bool reconfigure( double inSampleRate,double outSampleRate );

    void setDataIn( CF32 * fftin ); // this should not be called by the user app ... 
    
    double getOLASOutSampleRate();
    int getOLASOversampling();

    cudaEvent_t prepareTask(cudaEvent_t trigStart, cudaStream_t callerStream);
    void run();

    bool acceptsSamples();
    void setSleepingState( bool state );
    bool getSleepingState() ;

    void setCenterOfWindow( double freq );
    double getCenterOfWindow();
    void reset();

    CF32Block *get( int max_read=-1, int minimum_reply=16384);
    bool hasData( int max_read=-1 ) ; 


    cudaStream_t m_stream ;
    cudaEvent_t trigStop ;

private:
    bool m_sleeping ;
    double m_center_freq ;
    int NTaps ;
    double m_inSampleRate;
    int m_oversampling ;
    double m_outSampleRate;
    long data_size ;
    long wr_pos ;
    long out_wpos ;
    CF32 *datas_out;
    CF32* host_datas_gpu;

    CF32  *device_H, *_H0 ;
    CF32* data_in;
    CF32* device_data;
    CF32* device_decim_output ;


    int sumPaddingSize ;
    int num_SMs ;

    char *host_output_valid ;
    char *device_output_valid ;

    GPUChannelizer *device_channel ;
    GPUChannelizer *host_channel ;
    cufftHandle plan ;
    std::mutex *sem ; 

    static double bessi0( double x );
    static double *calc_filter(double Fs, double Fa, double Fb, int M, double Att, int *Ntaps ) ;
};

#endif // OLASGPUCHANNEL_H
