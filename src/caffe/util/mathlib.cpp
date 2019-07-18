#ifndef MATHLAB_H
#define MATHLAB_H

#define USE_MKL 1
#define USE_OPENBLAS 1

#if USE_MKL
#include <mkl.h>

#elif USE_OPENBLAS
#include <cblas.h>

#endif

#endif