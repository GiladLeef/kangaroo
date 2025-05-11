ifdef gpu

SRC = SECPK1/IntGroup.cpp Main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      GPU/GPUEngine.o Kangaroo.cpp HashTable.cpp \
      Backup.cpp Thread.cpp Check.cpp Network.cpp Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o Main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      GPU/GPUEngine.o Kangaroo.o HashTable.o Thread.o \
      Backup.o Check.o Network.o Merge.o PartMerge.o)

else

SRC = SECPK1/IntGroup.cpp Main.cpp SECPK1/Random.cpp \
      Timer.cpp SECPK1/Int.cpp SECPK1/IntMod.cpp \
      SECPK1/Point.cpp SECPK1/SECP256K1.cpp \
      Kangaroo.cpp HashTable.cpp Thread.cpp Check.cpp \
      Backup.cpp Network.cpp Merge.cpp PartMerge.cpp

OBJDIR = obj

OBJET = $(addprefix $(OBJDIR)/, \
      SECPK1/IntGroup.o Main.o SECPK1/Random.o \
      Timer.o SECPK1/Int.o SECPK1/IntMod.o \
      SECPK1/Point.o SECPK1/SECP256K1.o \
      Kangaroo.o HashTable.o Thread.o Check.o Backup.o \
      Network.o Merge.o PartMerge.o)

endif

CXX        = g++
CUDA       = /usr/local/cuda
CXXCUDA    = /usr/bin/g++
NVCC       = $(CUDA)/bin/nvcc
CXXFLAGS   =  -m64 -march=native -mtune=native -msse4.2 -mavx2 -ffast-math -funroll-loops -fomit-frame-pointer -flto -Wno-write-strings -Wno-unused-result -O3 -I.
ifdef gpu


CXXFLAGS   = -DWITHGPU -m64 -march=native -mtune=native -msse4.2 -mavx2 -ffast-math -funroll-loops -fomit-frame-pointer -flto -Wno-write-strings -Wno-unused-result -O3 -I. -I$(CUDA)/include
LFLAGS     = -lpthread -L$(CUDA)/lib64 -lcudart

else

CXXFLAGS   = -m64 -march=native -mtune=native -msse4.2 -mavx2 -ffast-math -funroll-loops -fomit-frame-pointer -flto -Wno-write-strings -Wno-unused-result -O3 -I.
LFLAGS     = -lpthread

endif

ifdef gpu
$(OBJDIR)/GPU/GPUEngine.o: GPU/GPUEngine.cu
	$(NVCC) -maxrregcount=0 --ptxas-options=-v --compile --compiler-options -fPIC -ccbin $(CXXCUDA) -m64 -O3 -I$(CUDA)/include -gencode=arch=compute_$(ccap),code=sm_$(ccap) -o $(OBJDIR)/GPU/GPUEngine.o -c GPU/GPUEngine.cu
endif
$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: Kangaroo

Kangaroo: $(OBJET)
	@echo Making Kangaroo...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

$(OBJET): | $(OBJDIR) $(OBJDIR)/SECPK1 $(OBJDIR)/GPU

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/SECPK1: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p SECPK1

$(OBJDIR)/GPU: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p GPU

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/SECPK1/*.o
	@rm -f obj/GPU/*.o