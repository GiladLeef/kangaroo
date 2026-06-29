OBJDIR = obj

SRC = math/cpu/intgroup.cpp main.cpp \
      core/timer.cpp math/cpu/int.cpp math/cpu/intmod.cpp \
      math/point.cpp math/secp256k1.cpp \
      core/kangaroo.cpp memory/hashtable.cpp core/thread.cpp \
      storage/check.cpp storage/backup.cpp core/network.cpp \
      storage/merge.cpp

OBJET = $(addprefix $(OBJDIR)/,$(SRC:.cpp=.o))

CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = g++
NVCC       = nvcc
ifdef gpu


CXXFLAGS   = -std=gnu++23 -DWITHGPU -m64 -march=native -mtune=native -msse4.2 -mavx2 -ffast-math -funroll-loops -fomit-frame-pointer -flto -Wno-write-strings -Wno-unused-result -O3 -I. -Icore -Imemory -Istorage -Imath -Imath/cpu -Imath/gpu -I$(CUDA)/include
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart
OBJET     += $(OBJDIR)/math/gpu/engine.o

else

CXXFLAGS   = -std=gnu++23 -m64 -march=native -mtune=native -msse4.2 -mavx2 -ffast-math -funroll-loops -fomit-frame-pointer -flto -Wno-write-strings -Wno-unused-result -O3 -I. -Icore -Imemory -Istorage -Imath -Imath/cpu -Imath/gpu
LFLAGS     = -lpthread

endif

ifdef gpu
$(OBJDIR)/math/gpu/engine.o: math/gpu/engine.cu
	@mkdir -p $(dir $@)
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O3 -I. -Icore -Imemory -Istorage -Imath -Imath/cpu -Imath/gpu -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/math/gpu/engine.o -c math/gpu/engine.cu
endif
$(OBJDIR)/%.o : %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: Kangaroo

Kangaroo: $(OBJET)
	@echo Making Kangaroo...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

clean:
	@echo Cleaning...
	@rm -rf $(OBJDIR)
