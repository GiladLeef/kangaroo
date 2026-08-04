# Kangaroo — plain header-based build (no modules), always with GPU.
# GPU arch auto-detected from nvidia-smi; override with:  make CCAP=86

CXX        = clang++
CUDA      ?= /usr/local/cuda

# ---- Compute capability (e.g. "61" for GTX 1080 Ti -> sm_61) ----
CCAP      ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
CCAP      ?= 61

# Cap device registers so ptxas leaves enough per SM for 6 blocks/SM occupancy.
# Without this cap clang's default allocation (~148 regs) drops occupancy to
# 3 blocks/SM, running ~30% slower than nvcc. ~80 regs restores nvcc-level speed.
GPU_MAXRREG ?= 80

OBJDIR    = obj

INCS      = -I$(CUDA)/include -I. -Icore -Imemory -Istorage -Imath -Imath/cpu -Imath/gpu -Icompat

CXXFLAGS  = -std=c++23 -m64 -march=native -mtune=native -msse4.2 -mavx2 \
            -ffast-math -funroll-loops -fomit-frame-pointer \
            -Wno-unused-result -Wno-write-strings -Wno-vla-cxx-extension -O3 $(INCS)

# Host builds emit the kernel launcher for CUDA-aware translation units.
CUDAFLAGS = $(CXXFLAGS) -x cuda --cuda-gpu-arch=sm_$(CCAP) --cuda-path=$(CUDA) \
            -Xcuda-ptxas -maxrregcount=$(GPU_MAXRREG)
LDFLAGS   = -lpthread -L$(CUDA)/lib64 -lcudart

SOURCES   = main.cpp math/cpu/int.cpp math/cpu/intmod.cpp math/cpu/intgroup.cpp \
            math/point.cpp math/secp256k1.cpp memory/hashtable.cpp core/timer.cpp \
            core/kangaroo.cpp core/thread.cpp core/network.cpp \
            storage/check.cpp storage/backup.cpp storage/merge.cpp
OBJECTS   = $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
OBJECTS  += $(OBJDIR)/math/gpu/engine.o

all: kangaroo

kangaroo: $(OBJECTS)
	@echo "Linking kangaroo..."
	$(CXX) $(OBJECTS) $(LDFLAGS) -o kangaroo

$(OBJDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

$(OBJDIR)/math/gpu/engine.o: math/gpu/engine.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CUDAFLAGS) -o $@ -c math/gpu/engine.cu

clean:
	@echo Cleaning...
	rm -rf $(OBJDIR) kangaroo

.PHONY: all clean