#if defined(__cplusplus) && defined(__CUDACC__)
#ifdef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#endif
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
#endif

#include_next <bits/mathcalls.h>
