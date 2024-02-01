/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#include "olasgpuchannel.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "fircoeffcache.h"

#define OLAS_DEBUG (0)
#define OLAS_TUNE_DEBUG (0)
#define OLAS_MATLAB_DEBUG (0)

int DDCBankChannel::FILTER_KERNEL_SIZE = 7200 ;

DDCBankChannel::DDCBankChannel(double inSampleRate, double outSampleRate, int oversampling, int maxInSize, unsigned int use_fft_size)
{
    int k ;
    cudaError_t rc ;
    datas_out =nullptr;
    host_datas_gpu =nullptr;
    device_H =nullptr;
    _H0 =nullptr;
    data_in =nullptr;
    device_data =nullptr;
    device_decim_output =nullptr;
    host_output_valid =nullptr;
    device_output_valid =nullptr;
    device_channel =nullptr;
    m_center_freq = 0 ;

    host_channel = (GPUChannelizer *)malloc(sizeof(GPUChannelizer));
    memset( host_channel, 0, sizeof(GPUChannelizer));
    sem = new std::mutex(); 
    //
    cudaEventCreate( &trigStop);

    m_inSampleRate = inSampleRate ;
    m_outSampleRate = outSampleRate ;
    m_oversampling = oversampling ;
    host_channel->fft_size = use_fft_size ;
    m_sleeping = true ;


    data_size = maxInSize; //(maxInSize + fft_size + NTaps );
    cufftPlan1d(&plan, host_channel->fft_size, CUFFT_C2C, 1) ;

    k = 2 ;
    while( m_inSampleRate/k > outSampleRate )
        k++ ;

    host_channel->DecimFactor = k-- ;
    m_outSampleRate = m_inSampleRate / host_channel->DecimFactor ;

    // allouer structure GPUChannelizer sur device
    rc = cudaMalloc((void **)&device_channel, sizeof(GPUChannelizer));
    assert( rc == cudaSuccess) ;

    // Do we already have that filter ?
    float cutoff = m_inSampleRate* 1.0/(host_channel->DecimFactor*2.0) ;
    FirCoeffCache& fcc = FirCoeffCache::getInstance() ;
    FIRCacheEntry *entry = fcc.get( cutoff, FILTER_KERNEL_SIZE );
    double *H =nullptr;
    if( entry ==nullptr) {
        H = calc_filter(m_inSampleRate,0, cutoff ,
                        FILTER_KERNEL_SIZE, 55.0, &NTaps) ;
        fcc.store( cutoff, FILTER_KERNEL_SIZE, NTaps, H );
    } else {
        H = entry->taps ;
        NTaps = entry->length ;
    }
    host_channel->overlap = NTaps-1 ;
    host_channel->decimated_sample_count  = ((use_fft_size-host_channel->overlap) * m_oversampling / host_channel->DecimFactor) + 1 ;
    host_channel->oversampling = m_oversampling ;

    _H0 = (CF32 *)cpxalloc( host_channel->fft_size );
    rc = cudaMalloc((void **)&device_H, sizeof(CF32) * host_channel->fft_size) ;
    assert( rc == cudaSuccess) ;

    // calculer la FFT du filtre
    // recopier coeffs du filtre dans entree FFT
    for( int i=0; i < NTaps ; i++ ) {
        _H0[i].I = (float)H[i] ;
        _H0[i].Q = (float)0 ;
    }
    // padder à 0 jusqu'a fft_size
    for( int i=NTaps; i < host_channel->fft_size ; i++ ) {
        _H0[i].I = (float)0 ;
        _H0[i].Q = (float)0 ;
    }

    if( 0 ) {
        FILE* f = fopen( "filter_taps.dat", "wb");
        fwrite( H, sizeof(double),NTaps,f);
        fclose(f);
    }
    //free(H);

    // calculer une bonne fois pour toute la FFT du filtre
    rc = cudaMemcpy(device_H, _H0, host_channel->fft_size*sizeof(CF32),  cudaMemcpyHostToDevice);
    assert( rc == cudaSuccess) ;
    cufftExecC2C(plan, (cufftComplex *)device_H,(cufftComplex *)device_H, CUFFT_FORWARD); // _H = fft(_H)
    // conserver la valeur
    rc = cudaMemcpy(_H0, device_H, host_channel->fft_size*sizeof(CF32), cudaMemcpyDeviceToHost) ;
    assert( rc == cudaSuccess) ;
    for( int i=0 ; i < host_channel->fft_size ; i++ ) {
        _H0[i].I *= 1.0/host_channel->fft_size ;
        _H0[i].Q *= 1.0/host_channel->fft_size ;
    }
    if( 0 ) {
        FILE* f = fopen( "/tmp/filter.cf32", "wb");
        fwrite( _H0, sizeof(CF32),host_channel->fft_size,f);
        fclose(f);
    }

    rc = cudaMalloc((void **)&device_data, sizeof(CF32) * host_channel->fft_size) ;
    assert( rc == cudaSuccess) ;
    rc = cudaMalloc((void **)&device_decim_output, sizeof(CF32) * host_channel->decimated_sample_count  ) ;
    assert( rc == cudaSuccess) ;
    rc = cudaMalloc((void **)&device_output_valid, sizeof(char) * host_channel->fft_size  ) ;
    assert( rc == cudaSuccess) ;

    datas_out = (CF32*)cpxalloc( data_size );
    host_datas_gpu = (CF32*)cpxalloc( host_channel->decimated_sample_count );
    host_output_valid = (char *)malloc( host_channel->decimated_sample_count  * sizeof(char)) ;
    memset( host_datas_gpu, 0, sizeof(CF32) * host_channel->decimated_sample_count ) ;
    memset( host_output_valid, 0, sizeof(char) * host_channel->decimated_sample_count ) ;

#ifndef __aarch64__
    rc = cudaHostRegister( host_datas_gpu, host_channel->decimated_sample_count  * sizeof(CF32), cudaHostRegisterPortable );
    assert( rc == cudaSuccess) ;
    rc = cudaHostRegister( host_output_valid, host_channel->decimated_sample_count  * sizeof(char), cudaHostRegisterPortable );
    assert( rc == cudaSuccess) ;
    rc = cudaHostRegister( host_channel, sizeof(GPUChannelizer), cudaHostRegisterPortable );
    assert( rc == cudaSuccess) ;
#endif

    out_wpos = 0 ;
    m_sleeping = true ;
    wr_pos = 0 ;
}

bool DDCBankChannel::reconfigure(double inSampleRate, double outSampleRate) {
    const std::lock_guard<std::mutex> lock(*sem);
    // reset
    out_wpos =  wr_pos = 0 ;
    host_channel->decim_r_pos = 0 ;
    wr_pos = 0 ;
    host_channel->mix_phase = 0 ;
    host_channel->mix_offset = 0 ;

    if( (fabs(inSampleRate-m_inSampleRate)<=10) && (fabs(outSampleRate-m_outSampleRate)<10)) {
        return( true );
    }

    m_inSampleRate = inSampleRate ;
    m_outSampleRate = outSampleRate ;

    int  k = 2 ;
    while( m_inSampleRate/k > outSampleRate )
        k++ ;
    host_channel->DecimFactor = k-- ;
    m_outSampleRate = m_inSampleRate / host_channel->DecimFactor ;
    // Do we already have that filter ?
    float cutoff = m_inSampleRate* 1.0/(host_channel->DecimFactor*2.0) ;
    FirCoeffCache& fcc = FirCoeffCache::getInstance() ;
    FIRCacheEntry *entry = fcc.get( cutoff, FILTER_KERNEL_SIZE );
    double *H =nullptr;
    if( entry ==nullptr) {
        H = calc_filter(m_inSampleRate,0, cutoff ,
                        FILTER_KERNEL_SIZE, 55.0, &NTaps) ;
        fcc.store( cutoff, FILTER_KERNEL_SIZE, NTaps, H );
    } else {
        H = entry->taps ;
        NTaps = entry->length ;
    }
    host_channel->overlap = NTaps-1 ;
    host_channel->decimated_sample_count  = (host_channel->fft_size-host_channel->overlap) / host_channel->DecimFactor + 1 ;
    // calculer la FFT du filtre
    // recopier coeffs du filtre dans entree FFT
    for( int i=0; i < NTaps ; i++ ) {
        _H0[i].I = (float)H[i] ;
        _H0[i].Q = (float)0 ;
    }
    // padder à 0 jusqu'a fft_size
    for( int i=NTaps; i < host_channel->fft_size ; i++ ) {
        _H0[i].I = (float)0 ;
        _H0[i].Q = (float)0 ;
    }
    // calculer une bonne fois pour toute la FFT du filtre
    cudaError_t rc = cudaMemcpy(device_H, _H0, host_channel->fft_size*sizeof(CF32),  cudaMemcpyHostToDevice);
    assert( rc == cudaSuccess) ;
    cufftExecC2C(plan, (cufftComplex *)device_H,(cufftComplex *)device_H, CUFFT_FORWARD); // _H = fft(_H)
    // conserver la valeur
    rc = cudaMemcpy(_H0, device_H, host_channel->fft_size*sizeof(CF32), cudaMemcpyDeviceToHost) ;
    assert( rc == cudaSuccess) ;
    for( int i=0 ; i < host_channel->fft_size ; i++ ) {
        _H0[i].I *= 1.0/host_channel->fft_size ;
        _H0[i].Q *= 1.0/host_channel->fft_size ;
    }
    m_center_freq = 0 ;
    m_sleeping = false ;
    return( true );
}

DDCBankChannel::~DDCBankChannel() {
#ifndef __aarch64__
    cudaHostUnregister(host_datas_gpu);
    cudaHostUnregister(host_output_valid);
    cudaHostUnregister(host_channel);
#endif
    cufftDestroy(plan);
    cpxfree(_H0);
    cudaFree(device_H);
    cudaFree(device_data);
    cudaFree(device_decim_output);
    cudaFree(device_output_valid);
    cudaFree(device_channel);
    cpxfree(datas_out);
    cpxfree(host_datas_gpu);
    free(host_output_valid);
    free(host_channel);
    delete sem ; 
    cudaEventDestroy( trigStop );
}

void DDCBankChannel::setSleepingState( bool state ) {
    m_sleeping = state ;
    if( OLAS_DEBUG ) {
        fprintf( stdout, "OLASGPUChannel::setSleepingState(%s)\n", m_sleeping ? "true":"false");
        fflush( stdout );
    }

}

bool DDCBankChannel::getSleepingState() {
    return(m_sleeping);
}

bool DDCBankChannel::acceptsSamples() {
    if( (out_wpos + host_channel->fft_size/host_channel->DecimFactor) > data_size ) {
        if( OLAS_DEBUG ) {
            fprintf( stdout, "OLASGPUChannel::step2() out full ?\n");
            fprintf( stdout, "OLASGPUChannel::step2() out_wpos= %ld\n" , out_wpos ) ;
            fprintf( stdout, "OLASGPUChannel::step2() host_channel->fft_size = %d\n",  host_channel->fft_size);
            fprintf( stdout, "OLASGPUChannel::step2() host_channel->fft_size= %d\n",  host_channel->fft_size );
            fprintf( stdout, "OLASGPUChannel::step2() host_channel->DecimFactor= %d\n",  host_channel->DecimFactor );
            fflush( stdout );
        }
        return(false) ;
    }
    return( !m_sleeping );
}

void DDCBankChannel::setDataIn( CF32* fftin) {
    data_in = fftin ;
    cufftSetStream( plan, m_stream );
}

double DDCBankChannel::getOLASOutSampleRate() {
    return( m_outSampleRate );
}

int DDCBankChannel::getOLASOversampling() {
    return( m_oversampling );
}

double DDCBankChannel::getCenterOfWindow() {
    return( m_center_freq );
}

inline float sign(float x) {
    if( x < 0 ) return(-1);
    return(1);
}

void DDCBankChannel::setCenterOfWindow(double freq ) {
    CF32 *tmp =nullptr; //[fft_size] ;
    CF32 *fftshift_of_H0 ;

    if( (fabs(freq) + m_outSampleRate/2.0) > m_inSampleRate/2.0) {
        double user_req = freq ;
        freq = sign(freq)*((m_inSampleRate-m_outSampleRate)/2.0 ) ;
        fprintf( stdout, "OLASGPUChannel::setCenterOfWindow() warning, freq outside window, fixed at = %f instead of %f\n" , freq , user_req);
        fflush(stdout);
    }

    const std::lock_guard<std::mutex> lock(*sem);
    if( OLAS_TUNE_DEBUG ) {
        fprintf( stdout, "OLASGPUChannel::setCenterOfWindow() freq = %f\n" , freq );
    }

    double bin_width = m_inSampleRate / host_channel->fft_size ;
    double diff = fabs( freq - m_center_freq );
    if( diff < bin_width ) {
        if( OLAS_TUNE_DEBUG ) {
            fprintf( stdout, "OLASGPUChannel::setCenterOfWindow() change too small\n");
        }
        return ;
    }



    if( fabs(freq) < bin_width ) {
        // no rotate the filter, shift too small. Just give back the original filter
        if( OLAS_TUNE_DEBUG ) {
            fprintf( stdout, "OLASGPUChannel::setCenterOfWindow() give back the original filter\n");
        }
        cudaMemcpy(device_H, _H0, host_channel->fft_size*sizeof(CF32),  cudaMemcpyHostToDevice);
        m_center_freq = 0 ;
        host_channel->mix_phase = 0 ;
        host_channel->mix_offset = 0 ;
        return ;
    }

    // filter must be rotated
    // reorder coefficients to natural order (-f/2...f/2) before rotating
    tmp = (CF32*)cpxalloc(host_channel->fft_size);
    fftshift_of_H0 = (CF32*)cpxalloc(host_channel->fft_size);
    memcpy( fftshift_of_H0, _H0, sizeof(CF32)*host_channel->fft_size);
    // fft shift the result
    {
        int n2 = host_channel->fft_size / 2 ;
        for( int i=0 ; i < n2 ; i++ ) {
            CF32 t = fftshift_of_H0[i];
            fftshift_of_H0[i] = fftshift_of_H0[i+n2] ;
            fftshift_of_H0[i+n2] = t ;
        }
    }

    m_center_freq = freq ;
    int decal = (int)floorf( freq * (host_channel->fft_size * 1.0/m_inSampleRate)  );
    host_channel->mix_phase = 0 ;
    host_channel->mix_offset = fmod( -freq/m_inSampleRate*2*M_PI , 2*M_PI );
    if( OLAS_TUNE_DEBUG ) fprintf( stdout, "OLASGPUChannel::setCenterOfWindow() shifting by %d bins\n", decal) ;

    for( int i=0 ; i < host_channel->fft_size ; i++ ) {
        int idx_dest = (i + decal) % host_channel->fft_size ;
        if( idx_dest < 0 ) {
            idx_dest = host_channel->fft_size + idx_dest ;
        }
        tmp[idx_dest].I = fftshift_of_H0[i].I;
        tmp[idx_dest].Q = fftshift_of_H0[i].Q;
    }

    if( OLAS_MATLAB_DEBUG ) {
        FILE* fw = fopen( "filter_taps_dec.dat", "wb");
        fwrite( tmp,sizeof(CF32), host_channel->fft_size,fw);
        fclose(fw);
    }
    //ifftshift the result
    {
        int n2 = host_channel->fft_size / 2 ;
        for( int i=n2 ; i < host_channel->fft_size ; i++ ) {
            CF32 t = tmp[i];
            tmp[i] = tmp[i-n2] ;
            tmp[i-n2] = t ;
        }
    }
    if( OLAS_MATLAB_DEBUG ) {
        FILE* fw = fopen( "filter_taps_dec_ifftshift.dat", "wb");
        fwrite( tmp,sizeof(CF32), host_channel->fft_size,fw);
        fclose(fw);
    }

    cudaMemcpy(device_H, tmp, host_channel->fft_size*sizeof(CF32),  cudaMemcpyHostToDevice);
    cpxfree( tmp) ;
    cpxfree( fftshift_of_H0 );
}

void DDCBankChannel::reset() {
    out_wpos =  wr_pos = 0 ;
    host_channel->decim_r_pos = 0 ;
    wr_pos = 0 ;
    host_channel->mix_phase = 0 ;
    host_channel->mix_offset = 0 ;
}

// faire le produit de Kroeneker entre la FFT du signal entrant
// et la FFT du filtre
/* revenir dans le domaine temporel et faire le mixage
 * Attention... on fait le mix pendant la décimation donc on doit
 * prendre les bonnes valeurs de l'OL aux bons instants (ceux où on décime)
 */
cudaEvent_t DDCBankChannel::prepareTask(cudaEvent_t trigStart , cudaStream_t callerStream) {
    cudaError_t rc ;
    (void)(callerStream);

    cudaStream_t targetStream = m_stream ;

    rc = cudaMemcpyAsync(device_channel, host_channel, sizeof(GPUChannelizer),  cudaMemcpyHostToDevice, targetStream);
    if( rc != cudaSuccess ) {
        fprintf( stderr, "cudaMemcpyAsync error %d\n", (int)rc);
        fflush(stderr);
    }
    assert( rc == cudaSuccess );

    cudaStreamWaitEvent( targetStream, trigStart, 0);
    // device_data = fft(filtre) .* fft( signal )
    GPUconvolve( data_in, device_H, device_data, host_channel->fft_size, targetStream);
    // device_data  = ifft( fft(filtre) .* fft( signal ) )
    cufftExecC2C(plan, (cufftComplex *)device_data, (cufftComplex *)device_data, CUFFT_INVERSE);
    cudaMemsetAsync( device_output_valid, 0 , sizeof(char) * host_channel->decimated_sample_count , targetStream ); // set sum to 0
    multiChannel( device_channel, host_channel->decimated_sample_count,
                  device_data, device_decim_output, device_output_valid, targetStream );

    //
    cudaMemcpyAsync( host_datas_gpu, device_decim_output, sizeof(CF32) * host_channel->decimated_sample_count , cudaMemcpyDeviceToHost, targetStream) ;
    cudaMemcpyAsync( host_output_valid, device_output_valid, sizeof(char) * host_channel->decimated_sample_count , cudaMemcpyDeviceToHost, targetStream) ;
    cudaEventRecord( trigStop, targetStream );
    return( trigStop );
}

void DDCBankChannel::run() {
    int p = 0;
    int q = 0 ;
    //double power = 0;
    CF32 tmp ;
    const std::lock_guard<std::mutex> lock(*sem);

    if( OLAS_DEBUG ) fprintf( stdout, "OLASGPUChannel::run() host_channel->decimated_sample_count=%d\n" , host_channel->decimated_sample_count );
    // A optimiser... pas idéal de faire une itération ici...
    //
    for( p=0 ; p < host_channel->decimated_sample_count ; p++ ) {
        char flag = host_output_valid[p] ;
        if( flag == (char)1) {
            tmp = host_datas_gpu[p] ;
            datas_out[out_wpos++] = tmp ;
            //host_channel->decim_r_pos += host_channel->DecimFactor  ;
            q++ ;
        }
    }
    host_channel->decim_r_pos += (q*host_channel->DecimFactor) / host_channel->oversampling ;
    host_channel->mix_phase += (host_channel->fft_size - host_channel->overlap) * host_channel->mix_offset ;
    //host_channel->mix_phase = fmod( host_channel->mix_phase, 2*M_PI ) ;

    p = floorf( fabsf(host_channel->mix_phase)/(2*M_PI)) ;
    if( p ) {
        q = (host_channel->mix_phase >= 0) ? -p:p;
        host_channel->mix_phase += q * 2 * M_PI ;
    }

    memset( host_datas_gpu, 0, sizeof(CF32) * host_channel->decimated_sample_count ) ;
 
}
 

CF32Block *DDCBankChannel::get(int max_read, int minimum_reply) {
    long remain = out_wpos ;
    if( (remain == 0 ) || (remain<minimum_reply)) {
        return(NULL);
    }
    const std::lock_guard<std::mutex> lock(*sem);
    if( max_read > 0 ) {
        if( max_read > remain ) {
            max_read = remain ;
        }
    } else {
        max_read = remain ;
    }


    CF32Block *result = CF32BlockAlloc(max_read);
    result->length    = max_read ;
    result->samplerate = m_outSampleRate ;
    result->center_freq = (int64_t)m_center_freq ;
    result->oversampling = m_oversampling ;

    memcpy( result->data, datas_out, max_read * sizeof(CF32));

    remain -= max_read ;
    if( remain > 0 ) {
        CF32* ptr = datas_out ;
        ptr += max_read ;
        memmove( (void *)datas_out, (void *)ptr, remain * sizeof(CF32));
        out_wpos = remain ;
    } else {
        out_wpos = 0 ;
    }
    if( OLAS_DEBUG ) fprintf( stdout, "OLASGPUChannel::get() max_read= %d , remain=%ld\n", max_read , remain ) ;

    return( result );
}


bool DDCBankChannel::hasData(int max_read) {
    const std::lock_guard<std::mutex> lock(*sem);
    long remain =  out_wpos ;
    if( remain == 0 ) {
        return(false);
    }

    if( max_read > 0 ) {
        return( max_read <= remain );
    }

    return(true);
}

//%  Fs=Sampling frequency
//%      * Fa=Low freq ideal cut off (0=low pass)
//%      * Fb=High freq ideal cut off (Fs/2=high pass)
//%      * Att=Minimum stop band attenuation (>21dB)
double *DDCBankChannel::calc_filter(double Fs, double Fa, double Fb, int M, double Att , int *Ntaps) {
    int k, j ;

    int len = 2*M + 1;
    double Alpha, C,x,y ;
    double *_H, *W ;

    _H = (double *)malloc( len * sizeof( double ));
    Alpha = .1102*(Att-8.7);

    // compute kaiser window of length 2M+1
    W = (double *)malloc( len * sizeof(double));
    assert( W !=nullptr) ;
    C = bessi0(Alpha*M_PI);

    for( k=0 ; k < len ; k++ ) {
        y = k*1.0/M - 1.0 ;
        x = M_PI*Alpha*sqrt( 1 - y*y) ;
        W[k] = bessi0(x)/C ;
        //printf("h(%d)=%0.8f;\n", k+1, W[k]);
    }

    k = 0 ;
    for( j=-M ; j <= M ; j++ ) {
        if( j == 0 ) {
            _H[k] = 2*(Fb-Fa)/Fs;
        } else {
            _H[k] = 1/(M_PI*j)*(sin(2*M_PI*j*Fb/Fs)-sin(2*M_PI*j*Fa/Fs))*W[k];
        }
        k++ ;
    }

    *Ntaps = len ;
    //    for( k=0 ; k < len ; k++ ) {
    //        printf("h(%d)=%0.8f;\n", k+1, H[k]*1000000);
    //    }
    free(W);
    return( _H );
}

double DDCBankChannel::bessi0( double x )
/*------------------------------------------------------------*/
/* PURPOSE: Evaluate modified Bessel function In(x) and n=0.  */
/*------------------------------------------------------------*/
{
    double ax,ans;
    double y;


    if ((ax=fabs(x)) < 3.75) {
        y=x/3.75,y=y*y;
        ans=1.0+y*(3.5156229+y*(3.0899424+y*(1.2067492
                                             +y*(0.2659732+y*(0.360768e-1+y*0.45813e-2)))));
    } else {
        y=3.75/ax;
        ans=(exp(ax)/sqrt(ax))*(0.39894228+y*(0.1328592e-1
                                              +y*(0.225319e-2+y*(-0.157565e-2+y*(0.916281e-2
                                                                                 +y*(-0.2057706e-1+y*(0.2635537e-1+y*(-0.1647633e-1
                                                                                                                      +y*0.392377e-2))))))));
    }
    return ans;
}

