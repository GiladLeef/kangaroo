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

CXX        = g++

CXXFLAGS   =  -static -static-libgcc -static-libstdc++ -m64 -mssse3  -Wno-write-strings -O3 -I.

ifeq ($(OS),Windows_NT)
    LFLAGS = -static -static-libgcc -static-libstdc++ -lpthread -lws2_32
else
    LFLAGS = -static -static-libgcc -static-libstdc++ -lpthread
endif


$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

all: Kangaroo

Kangaroo: $(OBJET)
	@echo Making Kangaroo...
	$(CXX) $(OBJET) $(LFLAGS) -o kangaroo

$(OBJET): | $(OBJDIR) $(OBJDIR)/SECPK1

$(OBJDIR):
	mkdir -p $(OBJDIR)


$(OBJDIR)/SECPK1: $(OBJDIR)
	cd $(OBJDIR) && mkdir -p SECPK1

clean:
	@echo Cleaning...
	@rm -f obj/*.o
	@rm -f obj/SECPK1/*.o
