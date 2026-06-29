#ifndef POINTH
#define POINTH

#include "int.h"

class Point {

public:

  Point();
  Point(Int *cx,Int *cy,Int *cz);
  Point(Int *cx, Int *cz);
  Point(const Point &p);
  ~Point();
  bool isZero();
  bool equals(const Point &p) const;
  void Set(const Point &p);
  void Set(Int *cx, Int *cy,Int *cz);
  void Clear();
  void Reduce();
  std::string toString();

  Int x;
  Int y;
  Int z;

};

#endif // POINTH
