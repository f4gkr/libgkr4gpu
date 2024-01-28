/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank - lib gkr4gpu                       *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <random>

#include "lib/olasgpu.h"

// Returns number of millisecs since 01/01/1970
uint64_t getTimeStamp() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    uint64_t ms = (uint64_t)tp.tv_sec * (uint64_t)1000 + ((uint64_t)tp.tv_usec) / 1000;
    return( ms );
}


#define NDDC_CHANNELS (2)

int main() {
    fprintf( stdout, "DDC example\n");
    fprintf( stdout, "Filters are using : %d taps\n", DDCBankChannel::FILTER_KERNEL_SIZE ) ;
    fflush( stdout );

    DDCBank* ddcbank = new DDCBank( 100e6, NDDC_CHANNELS, 512*1024, 512*1024 );
    // First channel : 2MHz wide +25 MHz
    int channel1 = ddcbank->allocChannel( 2e6 );
    ddcbank->setCenterOfWindow( channel1, 25e6 );
    ddcbank->enterActiveState( channel1); // this channel is enabled

    // Second channel: 5MHz de large at -10 MHz
    int channel2 = ddcbank->allocChannel( 5e6 );
    ddcbank->setCenterOfWindow( channel2, -10e6 );
    ddcbank->enterActiveState( channel2); // this channel is enabled

    // we send 256k complex samples, simulating a source providing 100 MHz 
    // IQ samples
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1, 1);
    CF32Block *in = CF32BlockAlloc( 256*1024 );
    for( int i=0 ; i < (int)in->buff_length ; i++ ) {
         in->data[i].I = dist(gen) ;
         in->data[i].Q = dist(gen) ;
         in->length++ ;
    }
    in->center_freq = 100e6 ;
    in->samplerate  = 100e6 ;

    uint64_t t0 = getTimeStamp();
    uint64_t samples = 0 ;
    int tick = 200 ;

for( ; ; ) {
 
         samples += in->length ;

         // inject data into the DDC
         int rc = ddcbank->put( in );

         if( rc == NO_FEEDER ) {
             fprintf( stderr, "No active channel, data ignored\n");
             fflush( stderr );
             continue ;
         }
         if( rc == INPUT_TOO_LONG ) {
             fprintf( stderr, "Read outputs first !!! too many data waiting\n");
             fflush( stderr );
             continue ;
         }

         if( rc == NEED_MORE_DATA ) {
             // the input buffer is not full yet, nothing processed at this stage
             continue ;
         }

         // Check if we have some outputs ready
         for( int c=0 ; c < NDDC_CHANNELS ; c++ ) {
              if( ddcbank->hasData( c, -1 ) ) {
                  // Data available here
                  CF32Block *output = ddcbank->get( c );
                  //fprintf( stdout, "Channel %d - %ld samples available\n", c, (long)output->length );
                  //fflush( stdout );
                  CF32BlockFree(output);
              }
         }
         tick-- ;
         if( tick == 0 ) {
             bool active = ddcbank->isActiveState( channel1 );
             if( active ) {
                 ddcbank->enterSleepingState( channel1 );
             } else {
                 ddcbank->enterActiveState( channel1);
             }

             tick = 200 ;
             double ksps = (double)samples / (getTimeStamp() - t0) ;
             fprintf( stdout, "perfs : %f Millions samples / sec\n", ksps / 1000.0 ); fflush(stdout);
             t0 = getTimeStamp();
             samples = 0 ;
         }
    }

    return(0);
}