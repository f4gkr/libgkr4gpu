/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#include "olasgpu.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#define OVSM_DEBUG (0)
//#define CONV_FFT_SIZE 4096


/**
 * @brief Construct a new DDCBank::DDCBank object - This class is the main interface to the GPU Channelizer
 * 
 * @param inSampleRateHz - input sample rate, unit must be consistent with allocChannel
 * @param max_channels - maximum number of channels you will need in your app
 * @param fft_size  - the fft size; depends on the number of cuda cores you have... start with 512x1024
 * @param maxInSize  - the maximum block size you will put in the channelizer , depends on your input source (sdr or file or ...)
 */
DDCBank::DDCBank(double inSampleRateHz, int max_channels,
                 int fft_size, int maxInSize )
{
    cudaError_t rc ;
    m_inSampleRate = inSampleRateHz ;
    m_maxChannels = max_channels ;
    m_fftSize = fft_size ;
    m_kernelSize = DDCBankChannel::FILTER_KERNEL_SIZE ;
    data_size = (maxInSize + fft_size + 2*m_kernelSize+1 );
    m_maxInSize = maxInSize ;
    m_writer_id = -1 ;
    m_lastChannelUsed = m_maxChannels ;
    m_timestamp = 0 ;

    datas_in = (CF32*)cpxalloc( data_size );

    channels = (DDCBankChannel **)malloc( m_maxChannels * sizeof(DDCBankChannel*)) ;
    for( int i=0 ; i < m_maxChannels ; i++ ) {
        channels[i] = nullptr ;
    }

    rc = cudaMalloc((void **)&dIn, sizeof(CF32) * m_fftSize) ;
    assert( rc == cudaSuccess) ;
    rc = cudaMalloc((void **)&dOut, sizeof(CF32) * m_fftSize) ;
    assert( rc == cudaSuccess) ;
    cufftPlan1d(&plan, m_fftSize, CUFFT_C2C, 1) ;

    cudaStreamCreate( &m_stream );
    cufftSetStream( plan, m_stream );

#ifndef __aarch64__
    // sur les architecture Jetson "bas de gamme" - TX1 / TX2 / Xavier NX au moins
    // cette fonctionnalité n'est pas bien supportée pour le moment
    // pour éviter les mauvaises surprises, c'est désactivé
    rc = cudaHostRegister( datas_in, data_size * sizeof(CF32), cudaHostRegisterPortable );
    assert( rc == cudaSuccess) ;
#endif
    // lock
    sem = new std::mutex();
    wr_pos = rd_pos = 0 ;
    cudaEventCreate( &trigStart);
    cudaEventCreate( &trigAllDone);
}


DDCBank::~DDCBank() {
#ifndef __aarch64__
    cudaHostUnregister(datas_in );
#endif
    cpxfree(datas_in);


    cudaFree( dIn );
    cudaFree( dOut );

    cudaStreamDestroy( m_stream );
    cufftDestroy( plan );

    delete sem ;

    for( int i=0 ; i < m_maxChannels ; i++ ) {
        if( channels[i] != nullptr ) {
            DDCBankChannel* osc = channels[i] ;
            channels[i] = nullptr ;
            delete osc ;
        }
    }

    free( channels );
    cudaEventDestroy( trigStart );
    cudaEventDestroy( trigAllDone );
}

/**
 * @brief reconfigure all the channels at once
 * 
 * @param inSampleRate 
 * @param newconfig 
 * @return true 
 * @return false 
 */
bool DDCBank::reconfigure(double inSampleRate, std::map<int, DDCConfiguration *> newconfig) {
    const std::lock_guard<std::mutex> lock(*sem);
    m_inSampleRate = inSampleRate ;
    m_writer_id =-1 ;
    wr_pos = rd_pos = 0 ;
    for( int i=0 ; i < m_maxChannels ; i++ ) {
        DDCBankChannel* ddc = channels[i] ;
        DDCConfiguration *new_parameters = newconfig[i] ;
        if( channels[i] != nullptr ) {
            if( (new_parameters != nullptr ) && ( new_parameters->enabled == true )){
                ddc->reconfigure( inSampleRate, new_parameters->outSampleRate ) ;
                ddc->setCenterOfWindow( new_parameters->centerFreq ) ;
                if( m_writer_id < 0 )
                    m_writer_id = i ;
                ddc->setSleepingState(false);
            } else {
                // Canal inutilisé pour le moment, mais on reconfigure avec une valeur par défaut
                ddc->reconfigure( inSampleRate, inSampleRate / 32 ) ;
                ddc->setSleepingState(true);
            }
        }
    }
    if( OVSM_DEBUG ) debug();
    return( true );
}

/**
 * @brief plots what is going on inside
 * 
 */
void DDCBank::debug() {
    char tmp[255] ;
    tmp[0] = '[' ;
    for( int i=0 ; i < m_maxChannels ; i++ ) {
        if( channels[i] == nullptr ) {
            tmp[i+1] = '_' ;
        } else {
            if( i == m_writer_id) {
                if( channels[i]->getSleepingState() ) {
                    tmp[i+1] = 'W' ;
                } else {
                    tmp[i+1] = 'F' ;
                }
            } else {
                if( channels[i]->getSleepingState() ) {
                    tmp[i+1] = 'S' ;
                } else {
                    tmp[i+1] = 'A' ;
                }
            }
        }
        tmp[i+2] = ']';
        tmp[i+3] = (char)0 ;
    }
    fprintf( stdout, " %s %d \n", tmp, m_lastChannelUsed ); fflush(stdout);
}

int DDCBank::getChannelCount() {
    return( m_maxChannels );
}

/**
 * @brief returns the number of active channels (reading the outputs)
 * 
 * @return int 
 */
int DDCBank::getActiveChannelCount() {
    int res = 0 ;
    const std::lock_guard<std::mutex> lock(*sem);
    for( int i=0 ; i < m_maxChannels ; i++ ) {
        if( channels[i] != nullptr ) {
            if( !channels[i]->getSleepingState() )
                res++ ;
        }
    }

    return( res );
}

/**
 * @brief Checks that we have room left (spare channels)
 * 
 * @return true 
 * @return false 
 */
bool DDCBank::hasSpareChannel() {
    bool res = false ;
    const std::lock_guard<std::mutex> lock(*sem);
    for( int i=0 ; i < m_maxChannels ; i++ ) {
        if( channels[i] == nullptr ) {
            res = true ;
            break ;
        }
    }
    if( OVSM_DEBUG ) debug();
    return( res );
}

/**
 * @brief Returns the offset of channel_id (in sample_rate unit) compared to center (0).
 * 
 * @param channel_id 
 * @return double 
 */
double DDCBank::getCenterOfWindow(int channel_id) {
    double result = 0 ;
    const std::lock_guard<std::mutex> lock(*sem);

    if( channel_id < 0 ) return(result); ;
    if( channel_id >= m_maxChannels ) return(result); ;

    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        result = ch->getCenterOfWindow() ;
    }

    return(result) ;
}

/**
 * @brief Tune the channel channel_id. 0 is the center freq
 * 
 * @param channel_id 
 * @param freq 
 */
void DDCBank::setCenterOfWindow(int channel_id, float freq) {
    const std::lock_guard<std::mutex> lock(*sem);
    if( channel_id < 0 ) return ;
    if( channel_id >= m_maxChannels ) return ;

    DDCBankChannel *ch = channels[channel_id];
    ch->setCenterOfWindow(freq);
}

/**
 * @brief Resets read/write pointers of the channel channel_id, so it is empty
 * 
 * @param channel_id 
 */
void DDCBank::reset(int channel_id) {
    const std::lock_guard<std::mutex> lock(*sem);
    if( channel_id < 0 ) return ;
    if( channel_id >= m_maxChannels ) return ;

    DDCBankChannel *ch = channels[channel_id];
    if( channel_id == m_writer_id ) {
        for( int i=0 ; i < m_maxChannels ; i++ ) {
            if(channels[i] != nullptr ) {
                channels[i]->reset();
            }
        }
    } else {
        ch->reset();
    }
    if( OVSM_DEBUG ) debug();
}


/**
 * @brief Create a new DDC channel
 * 
 * @param outSampleRate the rate you want, it should be an integer fraction of the input rate
 * @param oversampling  oversampling, integer
 * @return int from 0 to number of channels-1. Returns < 0 if no room left
 */
int DDCBank::allocChannel(double outSampleRate , int oversampling) {
    int idx = -1 ;
    const std::lock_guard<std::mutex> lock(*sem);
    for( int i=0 ; i < m_maxChannels ; i++ ) {
        if( channels[i] == nullptr ) {
            idx = i ;
            channels[idx] = new DDCBankChannel( m_inSampleRate, outSampleRate, oversampling, m_maxInSize , m_fftSize);
            cudaStreamCreate( &channels[idx]->m_stream ) ;
            channels[idx]->setDataIn(dOut);

            if( m_writer_id == -1 ) {
                m_writer_id = idx ;
            }
            break ;
        }
    }

    // find the max indice of the used channel to avoid loop on all
    m_lastChannelUsed = m_maxChannels ;
    for( int i=m_maxChannels-1 ; i>=0; i-- ) {
        m_lastChannelUsed = i ;
        if( channels[i] != nullptr ) {
            break ;
        }
    }

    if( OVSM_DEBUG ) debug();
    return( idx );
}

/**
 * @brief We do not need this channel anymore
 * 
 * @param channel_id 
 */
void DDCBank::releaseChannel( int channel_id ) {
    if( channel_id < 0 ) return ;
    if( channel_id >= m_maxChannels ) return ;

    const std::lock_guard<std::mutex> lock(*sem);
    DDCBankChannel *ch = channels[channel_id];

    if( ch != nullptr ) {
        cudaStreamDestroy( ch->m_stream );
        delete ch ;
        channels[channel_id] = nullptr ;

        if( channel_id == m_writer_id ) {
            m_writer_id = -1 ;

            // find another filler
            // search for active
            for( int i=0 ; i < m_maxChannels ; i++ ) {
                if( (channels[i] != nullptr ) && (!channels[i]->getSleepingState())) {
                    m_writer_id = i ;
                    break ;
                }
            }
            // else not null
            if( m_writer_id == -1 ) {
                for( int i=0 ; i < m_maxChannels ; i++ ) {
                    if( channels[i] != nullptr  ) {
                        m_writer_id = i ;
                        break ;
                    }
                }
            }
        }
    }

    // find the max indice of the used channel to avoid loop on all
    m_lastChannelUsed = m_maxChannels ;
    for( int i=m_maxChannels-1 ; i>=0; i-- ) {
        m_lastChannelUsed = i ;
        if( channels[i] != nullptr ) {
            break ;
        }
    }

    if( OVSM_DEBUG ) debug();
}

/**
 * @brief We suspend this channel from the process, we do not need the output for a while, so we put it to sleep mode
 * 
 * @param channel_id 
 */
void DDCBank::enterSleepingState( int channel_id ) {
    const std::lock_guard<std::mutex> lock(*sem);
    if( channel_id < 0 ) return ;
    if( channel_id >= m_maxChannels ) return ;

    DDCBankChannel *ch = channels[channel_id];

    if( ch != nullptr ) {
        if( ch->getSleepingState() ) {
            if( OVSM_DEBUG ) debug();
            return ;
        }
        ch->setSleepingState(true);
        if( channel_id == m_writer_id ) {
            m_writer_id = -1 ;
            // find another filler
            for( int i=0 ; i < m_maxChannels ; i++ ) {
                if( ( channels[i] != NULL) && (i!=channel_id) && !channels[i]->getSleepingState()   ) {
                    m_writer_id = i ;
                    break ;
                }
            }
        }
    }
    if( OVSM_DEBUG ) debug();
}

/**
 * @brief returns true if the channel is in sleep mode
 * 
 * @param channel_id 
 * @return true 
 * @return false 
 */
bool DDCBank::isActiveState(int channel_id) {
    const std::lock_guard<std::mutex> lock(*sem);
    bool result = false ;
    if( channel_id < 0 ) return(false) ;
    if( channel_id >= m_maxChannels ) return(false) ;
    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        result = !ch->getSleepingState() ;
    }
    return( result );
}

void DDCBank::enterActiveState( int channel_id ) {
    const std::lock_guard<std::mutex> lock(*sem);
    if( channel_id < 0 ) return ;
    if( channel_id >= m_maxChannels ) return ;


    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        ch->setSleepingState(false);
        if( -1 == m_writer_id ) {
            m_writer_id = channel_id ;
        }
        // by default channel 0 is the filler
        if( channel_id == 0 ) {
            m_writer_id = 0 ;
        }

        if( channels[m_writer_id] != nullptr ) {
            if( channels[m_writer_id]->getSleepingState() ) {
                m_writer_id = channel_id ;
            }
        }

    }
    if( OVSM_DEBUG ) debug();
}



bool DDCBank::amIFeeder(int channel_id) {
    const std::lock_guard<std::mutex> lock(*sem);
    return( m_writer_id == channel_id );
}


int DDCBank::put(CF32 *src, long data_length ) {
    long buff_len = 0 ;
    long end = 0 ;

    if( src == nullptr )
        return(INPUT_TOO_LONG);
    end = wr_pos + data_length ;

    if( m_writer_id == -1 )
        return(NO_FEEDER); // no active channel ?

    if( end >= data_size ) {
        return(INPUT_TOO_LONG); // too long
    }


    ttl_put += data_length ;
    CF32 *ptw = datas_in ;
    ptw += wr_pos ;
    memcpy( (void *)ptw, src, data_length * sizeof(CF32));
    wr_pos = end ;
    buff_len = wr_pos - rd_pos ;
    if( buff_len < m_fftSize ) {
        return(NEED_MORE_DATA);
    }

    step1();

    return(GET_DATA_OUT);
}


int DDCBank::put(CF32Block *block) {
    if( block->oversampling > 1 ) {
        fprintf( stderr, "Oversampling non géré en entrée\n");
        fflush( stderr );
        assert( block->oversampling == 1 );
    }
    return( put( block->data, block->length ));
}
 

CF32Block *DDCBank::get(int channel_id, int max_read ) {
    const std::lock_guard<std::mutex> lock(*sem);
    CF32Block* result = nullptr ;

    if( channel_id < 0 ) return(NULL);
    if( channel_id >= m_maxChannels ) return(NULL);

    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        result = ch->get( max_read, 0 );
    }
    return(result) ;
}



double DDCBank::getOLASOutSampleRate( int channel_id ) {
    const std::lock_guard<std::mutex> lock(*sem);
    double result = -1 ;

    if( channel_id < 0 ) return(result); ;
    if( channel_id >= m_maxChannels ) return(result); ;

    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        result = ch->getOLASOutSampleRate() ;
    }

    return(result) ;
}



int DDCBank::getOLASOutOversampling(int channel_id) {
    const std::lock_guard<std::mutex> lock(*sem);
    int result = -1 ;

    if( channel_id < 0 ) return(result); ;
    if( channel_id >= m_maxChannels ) return(result); ;

    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        result = ch->getOLASOversampling() ;
    }

    return(result) ;
}


// Faire la FFT d'un bloc de signal d'entree
// puis avancer de la longeur de la FFT - la longeur du filtre
void DDCBank::step1() {
    const std::lock_guard<std::mutex> lock(*sem);
    int processed ;
    long buff_len = wr_pos - rd_pos ;
    CF32* r = nullptr ;
    int mem_size = sizeof(CF32) * m_fftSize;
    cudaError_t rc ;

    while( buff_len >= m_fftSize) {

        // recopier dans buffer d'entree FFT les données temporelle 'dIn'
        rc = cudaMemcpyAsync(dIn, datas_in, mem_size,  cudaMemcpyHostToDevice, m_stream);
        assert( rc == cudaSuccess );
        // calculer la FFT du signal
        cufftResult cur = cufftExecC2C(plan, (cufftComplex *)dIn, (cufftComplex *)dOut, CUFFT_FORWARD);
        assert( cur == CUFFT_SUCCESS );

        rc = cudaEventRecord(trigStart, m_stream );
        if( rc != cudaSuccess ) {
            fprintf( stderr, "CUDA Panic %d in %s\n", (int)rc, __func__ ) ;
        }
        assert( rc == cudaSuccess) ;

        processed = 0 ;
        for( int i=0 ; i < m_maxChannels ; i++ ) {
            if( channels[i] != nullptr ) {
                if( channels[i]->acceptsSamples() ) {
                    cudaEvent_t endTask = channels[i]->prepareTask(trigStart, m_stream);
                    rc = cudaStreamWaitEvent( m_stream, endTask, 0);
                    assert( rc == cudaSuccess) ;
                    processed++ ;
                }
            }
        }
        cudaEventRecord(trigAllDone, m_stream );


        // repositionner le buffer interne : décalage
        buff_len = wr_pos - m_fftSize + (2*m_kernelSize) ;
        r = datas_in + m_fftSize - (2*m_kernelSize) ;
        if( buff_len > 0 )
            memmove( (void *)datas_in, (void *)r, buff_len*sizeof(CF32));
        rd_pos = 0 ;
        wr_pos = buff_len ;

        if( (processed > 0) && (cudaStreamQuery( m_stream ) != cudaSuccess) ) {
            rc = cudaStreamSynchronize( m_stream );
            if( rc != cudaSuccess) {
                fprintf( stderr, "cudaStreamSynchronize rc=%d\n", (int)rc ) ;
            }
            assert( rc == cudaSuccess);
        }


        if( processed == 0 ) {
            m_writer_id = -1 ;
        } else {
            for( int i=0 ; i < m_maxChannels ; i++ ) {
                if( channels[i] != nullptr ) {
                    if( channels[i]->acceptsSamples() ) {
                        channels[i]->run(); // Cette partie pourrait être // via des Threads....
                    }
                }
            }
        }
    }
}


bool DDCBank::hasData(int channel_id, int max_read) {
    const std::lock_guard<std::mutex> lock(*sem);
    bool result = false ;

    if( channel_id < 0 ) return(result); ;
    if( channel_id >= m_maxChannels ) return(result); ;

    DDCBankChannel *ch = channels[channel_id];
    if( ch != nullptr ) {
        result = ch->hasData(max_read) ;
    }
    return(result) ;
}
