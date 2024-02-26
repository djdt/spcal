#ifndef _SPCAL_SORT_
#define _SPCAL_SORT_

#include <stdlib.h>

struct argsort {
  double value;
  size_t index;
};

void mergesort_argsort(struct argsort *x, size_t n);
void quicksort_argsort(struct argsort *x, size_t n);

#endif
