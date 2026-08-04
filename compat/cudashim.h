#pragma once

#include <features.h>

#ifdef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#undef __GLIBC_USE_IEC_60559_FUNCS_EXT_C23
#endif
#define __GLIBC_USE_IEC_60559_FUNCS_EXT_C23 0
