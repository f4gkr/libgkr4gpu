/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#include "fircoeffcache.h"
#include <string.h>

FirCoeffCache::FirCoeffCache()
{

}

FirCoeffCache::~FirCoeffCache() {
    std::map<std::string, FIRCacheEntry *>::iterator it ;
    for( it=fircache.begin() ; it != fircache.end() ; it++ ) {
         FIRCacheEntry *e = it->second ;
         free( e->taps );
         free(e);
    }
    fircache.clear();
}

FIRCacheEntry *FirCoeffCache::get(float cutoff, int order) {
    FIRCacheEntry *result = NULL ;
    int prct = (int)(cutoff*1000);
    std::string key = std::to_string(prct) + ":" + std::to_string(order);
    mtx.lock();
    if( fircache[key] != NULL ) {
        result = fircache[key] ;
    }
    mtx.unlock();
    return( result );
}

void FirCoeffCache::store(float cutoff, int order, int length, double *taps) {
    mtx.lock();
    int prct = (int)(cutoff*1000);
    std::string key = std::to_string(prct) + ":" + std::to_string(order);
    FIRCacheEntry *entry = (FIRCacheEntry *)malloc( sizeof(FIRCacheEntry));
    entry->cutoff = cutoff ;
    entry->length = length ;
    entry->taps = (double *)malloc( length * sizeof(double));
    memcpy( entry->taps, taps, length * sizeof(double));
    fircache[key] = entry ;
    mtx.unlock();
}
