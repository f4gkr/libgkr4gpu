/****************************************************************
 *                                                              *
 * @copyright  Copyright (c) Sylvain Azarian - F4GKR            *
 * @author     Sylvain AZARIAN - s.azarian@sdr-technologies.fr  *
 * @project    GPU DDC Bank                                     *
 *                                                              *
 * Licence: GPL 3.0                                             *
 *                                                              *
 ****************************************************************/

#ifndef FIRCOEFFCACHE_H
#define FIRCOEFFCACHE_H

#include <map>
#include <mutex>
#include "datatypes.h"

typedef struct {
    float cutoff ;
    int   length ;
    double *taps ;
} FIRCacheEntry ;



class FirCoeffCache
{
public:
    static FirCoeffCache& getInstance()  {
            static FirCoeffCache instance;
            return instance;
    }

    FIRCacheEntry *get( float cutoff, int order );
    void store( float cutoff, int order, int length, double *taps );

private:
    std::mutex mtx ;
    std::map<std::string, FIRCacheEntry*> fircache ;

    FirCoeffCache();
    FirCoeffCache(const FirCoeffCache &); // hide copy constructor
    FirCoeffCache& operator=(const FirCoeffCache &);
    ~FirCoeffCache();
};

#endif // FIRCOEFFCACHE_H
