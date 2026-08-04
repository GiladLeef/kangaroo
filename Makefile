# Kangaroo — built entirely with clang (host + CUDA device), always with GPU.
# GPU arch auto-detected from nvidia-smi; override with:  make CCAP=86

CXX        = clang++
CUDA      ?= /usr/local/cuda

# ---- Compute capability (e.g. "61" for GTX 1080 Ti -> sm_61) ----
CCAP      ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
CCAP      ?= 61

OBJDIR    = obj

INCS      = -I$(CUDA)/include -I. -Icore -Imemory -Istorage -Imath -Imath/cpu -Imath/gpu -Icompat

CXXFLAGS  = -std=c++23 -m64 -march=native -mtune=native -msse4.2 -mavx2 \
            -ffast-math -funroll-loops -fomit-frame-pointer -flto \
            -Wno-unused-result -Wno-write-strings -O3 $(INCS)

# Host builds emit the kernel launcher for CUDA-aware translation units.
CUDAFLAGS = $(CXXFLAGS) -x cuda --cuda-gpu-arch=sm_$(CCAP) --cuda-path=$(CUDA)
LDFLAGS   = -lpthread -L$(CUDA)/lib64 -lcudart -flto

SRC      = math/cpu/int.cpp math/cpu/intgroup.cpp math/cpu/intmod.cpp \
           math/point.cpp math/secp256k1.cpp \
           core/timer.cpp core/kangaroo.cpp core/thread.cpp core/network.cpp \
           memory/hashtable.cpp \
           storage/check.cpp storage/backup.cpp storage/merge.cpp \
           main.cpp

OBJET    = $(addprefix $(OBJDIR)/,$(SRC:.cpp=.o))
OBJET   += $(OBJDIR)/math/gpu/engine.o

all: Kangaroo

Kangaroo: $(OBJET)
	@echo "Linking Kangaroo..."
	$(CXX) $(OBJET) $(LDFLAGS) -o kangaroo

$(OBJDIR)/math/gpu/engine.o: math/gpu/engine.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CUDAFLAGS) -o $@ -c math/gpu/engine.cu

$(OBJDIR)/%.o : %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

clean:
	@echo Cleaning...
	rm -rf $(OBJDIR) kangaroo

.PHONY: all clean