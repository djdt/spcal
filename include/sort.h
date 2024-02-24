#ifndef _SPCAL_SORT_
#define _SPCAL_SORT_

struct argsort {
  double value;
  int index;
};

void mergesort_argsort(struct argsort *x, int n);
void quicksort_argsort(struct argsort *x, int n);

#endif
