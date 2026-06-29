#ifndef INTGROUPH
#define INTGROUPH

#include "int.h"
#include <vector>

class IntGroup {

public:

	IntGroup(int size);
	~IntGroup();
	void Set(Int *pts);
	void ModInv();

private:

	Int *ints;
  std::vector<Int> subp;
  int size;

};

#endif // INTGROUPCPUH
