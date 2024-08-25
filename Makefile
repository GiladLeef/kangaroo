# Compiler and Tools
CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc

# Flags
ifdef gpu
CXXFLAGS   = -DWITHGPU -m64 -mssse3 -O3 -I. -I$(CUDA)/include
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart
else
CXXFLAGS   = -m64 -march=native -mtune=native -mssse3 -pthread -ftree-vectorize -flto -O3 -funroll-loops -finline-functions -I.
LFLAGS     = -lpthread
endif

# Source and Object Files
SRC = SECPK1/IntGroup.cpp Main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
      Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

# GPU-specific Source
ifdef gpu
SRC += GPU/GPUEngine.cu
OBJDIR = obj
OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o Main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      GPU/GPUEngine.o Kangaroo.o HashTable.o Thread.o \
      Backup.o Check.o Network.o Merge.o PartMerge.o)
else
OBJDIR = obj
OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      Kangaroo.o HashTable.o Thread.o Check.o Backup.o \
      Network.o Merge.o PartMerge.o)
endif

# Build Targets
all: $(OBJDIR) kangaroo

kangaroo: $(OBJET)
	@echo "Making kangaroo..."
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

# Object File Rules
$(OBJDIR)/%.o: %.cpp | $(OBJDIR) $(OBJDIR)/SECPK1
	$(CXX) $(CXXFLAGS) -o $@ -c $<

ifdef gpu
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu | $(OBJDIR)/GPU
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O3 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $@ -c $<
endif

# Directory Rules
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/GPU: $(OBJDIR)
	mkdir -p $(OBJDIR)/GPU

$(OBJDIR)/SECPK1: $(OBJDIR)
	mkdir -p $(OBJDIR)/SECPK1

# Clean Up
clean:
	@echo "Cleaning..."
	@rm -f kangaroo
	@rm -f $(OBJDIR)/*.o
	@rm -f $(OBJDIR)/GPU/*.o
	@rm -f $(OBJDIR)/SECPK1/*.o

.PHONY: all clean
