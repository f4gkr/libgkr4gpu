/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#ifndef DATATYPES_H
#define DATATYPES_H

#include <stdlib.h>
#include <stdint.h>

typedef struct {
    float I;
    float Q;
} CF32 ;

typedef struct  {
    int16_t i;
    int16_t q;
} CS16 ;


#define MALLOC_V4SF_ALIGNMENT 64 // with a 64-byte alignment, we are even aligned on L2 cache lines...
inline CF32* cpxalloc( int nb_samples ) {
    CF32* r = (CF32*)aligned_alloc(MALLOC_V4SF_ALIGNMENT,nb_samples*sizeof(CF32)) ;
    return( r );
}

inline void cpxfree(void *p) {
    free(p) ;
}

typedef struct {
    CF32 *data ;            // pointer to the IQ samples
    uint32_t length ;       // the effective number of samples
    uint32_t buff_length ;  // the lenght of allocated region to store samples. buff_length >= length
    int64_t center_freq ;   // This is the offset of the channel
    uint64_t samplerate ;   // This is the sample rate of the channel
    uint8_t oversampling;   // oversampling factor 
} CF32Block ;


// Allocates a block of IQ Samples
inline CF32Block* CF32BlockAlloc( int length ) {

    CF32Block* res = (CF32Block*)malloc( sizeof( CF32Block ));
    if( res == nullptr )
        return( nullptr );

    res->buff_length = length ;
    res->length = 0 ; 
    res->center_freq = 0 ;
    res->samplerate = 0 ; 
    res->oversampling = 1 ;

    res->data = cpxalloc( length );
    if( res->data == nullptr ) {
        free( res );
        res = nullptr ;
    }

    return( res );
}

inline void CF32BlockFree( CF32Block* b ) {
    if( b == nullptr )
        return ;

    if( b->data != nullptr ) {
        cpxfree(b->data);
    }
    free(b);
}


#endif // DATATYPES_H
