/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#ifndef OLASGPU_H
#define OLASGPU_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <map>
#include "datatypes.h"

#include "olasgpuchannel.h"

class DDCConfiguration {
public:
    double outSampleRate ;
    double centerFreq ;
    bool   enabled ;
};

/***
 * This is you main interface to the GPU DDC Bank
*/

class DDCBank
{
public:

    
    explicit DDCBank(double inSampleRateHz, // your input sample rate. The unit must be consistent with the one you use when using AllocChannel
                     int max_channels,      
                     int fft_size,
                     int maxInSize);

    ~DDCBank();

    // reconfigure the DDC with new in sample rate and new channel configuration
    //
    bool reconfigure(double inSampleRate , std::map<int, DDCConfiguration *> newconfig);

    int getChannelCount();
    int getActiveChannelCount();


    int put( CF32 *src, long data_length ) ;
    int put( CF32Block *block );

    CF32Block* get(int channel_id, int max_read=-1);
    bool hasData( int channel_id, int max_read ); 

    double getOLASOutSampleRate( int channel_id );
    int getOLASOutOversampling( int channel_id );

    double getCenterOfWindow(int channel_id);
    void setCenterOfWindow(int channel_id,  float freq );

    void reset(int channel_id);

    // Channel allocation
    bool hasSpareChannel();
    int allocChannel(double outSampleRate , int oversampling=1);
    void releaseChannel( int channel_id );

    // Channel enable/disable
    bool isActiveState( int channel_id );
    void enterSleepingState( int channel_id );
    void enterActiveState( int channel_id );
    
 
private:
    cudaStream_t m_stream ;
    double m_timestamp ;
    double m_inSampleRate ;
    int m_maxChannels ;
    int m_lastChannelUsed ;
    int m_kernelSize ;
    int data_size ;
    int m_maxInSize ;
    int m_writer_id ;

    DDCBankChannel **channels ;
    cudaEvent_t trigStart ;
    cudaEvent_t trigAllDone ;

    long wr_pos ;
    long rd_pos ;
    long ttl_put ;

    CF32 *datas_in ;
    CF32 *dIn ;
    CF32 *dOut;
    cufftHandle plan ;

    int m_fftSize ;
    std::mutex *sem ;

    void step1();
    void debug();
    bool amIFeeder( int channel_id );
};

#endif // OLASGPU_H
