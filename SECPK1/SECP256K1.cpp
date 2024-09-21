#include "SECP256k1.h"
#include "IntGroup.h"
#include <string.h>

Secp256K1::Secp256K1() {}

void Secp256K1::Init() {
  // Prime for the finite field
  Int P;
  P.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");

  // Set up field
  Int::SetupField(&P);

  // Generator point and order
  G.x.SetBase16("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  G.y.SetBase16("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  G.z.SetInt32(1);
  order.SetBase16("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");

  Int::InitK1(&order);

  // Compute Generator table
  Point N(G);
  for(int i = 0; i < 32; i++) {
    GTable[i * 256] = N;
    N = DoubleDirect(N);
    for (int j = 1; j < 255; j++) {
      GTable[i * 256 + j] = N;
      N = AddDirect(N, GTable[i * 256]);
    }
    GTable[i * 256 + 255] = N; // Dummy point for check function
  }
}

Secp256K1::~Secp256K1() {}

Point Secp256K1::ComputePublicKey(Int *privKey,bool reduce) {
  int i = 0;
  Point Q;
  Q.Clear();

  for (; i < 32; i++) {
      uint8_t b = privKey->GetByte(i);
      if (b) {
          Q = GTable[256 * i + (b - 1)];
          i++;
          break;
      }
  }

  for (; i < 32; i++) {
      uint8_t b = privKey->GetByte(i);
      if (b)
          Q = Add2(Q, GTable[256 * i + (b - 1)]);
  }

  if(reduce) Q.Reduce();
  return Q;
}

std::vector<Point> Secp256K1::ComputePublicKeys(std::vector<Int> &privKeys) {
    std::vector<Point> pts;
    IntGroup grp((int)privKeys.size());
    Int *inv = new Int[privKeys.size()];
    pts.reserve(privKeys.size());

    for(int i = 0; i < privKeys.size(); i++) {
        Point P = ComputePublicKey(&privKeys[i], false);
        inv[i].Set(&P.z);
        pts.push_back(P);
    }

    grp.Set(inv);
    grp.ModInv();

    for(int i = 0; i < privKeys.size(); i++) {
        pts[i].x.ModMulK1(inv + i);
        pts[i].y.ModMulK1(inv + i);
        pts[i].z.SetInt32(1);
    }

    delete[] inv;
    return pts;
}

Point Secp256K1::NextKey(Point &key) {
  // Input key must be reduced and different from G
  // in order to use AddDirect
  return AddDirect(key,G);
}

uint8_t Secp256K1::GetByte(std::string &str, int idx) {
  char tmp[3];
  int  val;
  tmp[0] = str.data()[2 * idx];
  tmp[1] = str.data()[2 * idx + 1];
  tmp[2] = 0;

  if (sscanf(tmp, "%X", &val) != 1) {
    printf("ParsePublicKeyHex: Error invalid public key specified (unexpected hexadecimal digit)\n");
    exit(-1);
  }

  return (uint8_t)val;

}

bool Secp256K1::ParsePublicKeyHex(std::string str,Point &ret,bool &isCompressed) {
  ret.Clear();
  if (str.length() < 2) {
    printf("ParsePublicKeyHex: Error invalid public key specified (66 or 130 character length)\n");
    return false;
  }

  uint8_t type = GetByte(str, 0);

  switch (type) {

    case 0x02:
      if (str.length() != 66) {
        printf("ParsePublicKeyHex: Error invalid public key specified (66 character length)\n");
        return false;
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      ret.y = GetY(ret.x, true);
      isCompressed = true;
      break;

    case 0x03:
      if (str.length() != 66) {
        printf("ParsePublicKeyHex: Error invalid public key specified (66 character length)\n");
        return false;
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      ret.y = GetY(ret.x, false);
      isCompressed = true;
      break;

    case 0x04:
      if (str.length() != 130) {
        printf("ParsePublicKeyHex: Error invalid public key specified (130 character length)\n");
        exit(-1);
      }
      for (int i = 0; i < 32; i++)
        ret.x.SetByte(31 - i, GetByte(str, i + 1));
      for (int i = 0; i < 32; i++)
        ret.y.SetByte(31 - i, GetByte(str, i + 33));
      isCompressed = false;
      break;

    default:
      printf("ParsePublicKeyHex: Error invalid public key specified (Unexpected prefix (only 02,03 or 04 allowed)\n");
      return false;
  }

  ret.z.SetInt32(1);

  if (!EC(ret)) {
    printf("ParsePublicKeyHex: Error invalid public key specified (Not lie on elliptic curve)\n");
    return false;
  }

  return true;

}

std::string Secp256K1::GetPublicKeyHex(bool compressed, Point &pubKey) {
  unsigned char publicKeyBytes[128];
  char tmp[3];
  std::string ret;

  if (!compressed) {
    // Full public key
    publicKeyBytes[0] = 0x4;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);
    pubKey.y.Get32Bytes(publicKeyBytes + 33);

    for (int i = 0; i < 65; i++) {
      sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
      ret.append(tmp);
    }
  } else {
    // Compressed public key
    publicKeyBytes[0] = pubKey.y.IsEven() ? 0x2 : 0x3;
    pubKey.x.Get32Bytes(publicKeyBytes + 1);

    for (int i = 0; i < 33; i++) {
      sprintf(tmp, "%02X", (int)publicKeyBytes[i]);
      ret.append(tmp);
    }

  }
  return ret;
}
Point Secp256K1::AddDirect(Point &p1, Point &p2) {
    Int dy, dx, s, p, temp;
    Point r;
    r.z.SetInt32(1);

    dy.ModSub(&p2.y, &p1.y);
    dx.ModSub(&p2.x, &p1.x);
    dx.ModInv();  // dx = inverse(dx)

    s.ModMulK1(&dy, &dx);
    p.ModSquareK1(&s);

    r.x.ModSub(&p, &p1.x);
    r.x.ModSub(&p2.x);

    temp.ModSub(&p2.x, &r.x);
    temp.ModMulK1(&s, &temp);
    r.y.ModSub(&temp, &p2.y);

    return r;
}
std::vector<Point> Secp256K1::AddDirect(std::vector<Point> &p1, std::vector<Point> &p2) {
    if (p1.size() != p2.size()) {
        printf("Secp256K1::AddDirect: vectors have not the same size\n");
        exit(-1);
    }

    int size = static_cast<int>(p1.size());
    std::vector<Point> pts;
    pts.reserve(size);

    std::vector<Int> dx(size), dy(size);
    IntGroup grp(size);

    for (int i = 0; i < size; ++i) {
        dx[i].ModSub(&p2[i].x, &p1[i].x);
    }
    grp.Set(dx.data());
    grp.ModInv();

    for (int i = 0; i < size; ++i) {
        Point r;
        r.z.SetInt32(1);

        if (p1[i].x.IsZero()) {
            pts.push_back(p2[i]);
        } else {
            dy[i].ModSub(&p2[i].y, &p1[i].y);

            Int s, p;
            s.ModMulK1(&dy[i], &dx[i]);
            p.ModSquareK1(&s);

            r.x.ModSub(&p, &p1[i].x);
            r.x.ModSub(&p2[i].x);

            Int temp;
            temp.ModSub(&p2[i].x, &r.x);
            temp.ModMulK1(&s, &temp);
            r.y.ModSub(&temp, &p2[i].y);

            pts.push_back(r);
        }
    }
    return pts;
}

Point Secp256K1::Add2(Point &p1, Point &p2) {
    // Ensure p2.z = 1 for the operation
    Int u, v, u1, v1, us2, vs2, vs3, us2w, vs2v2, vs3u2, a;
    Point r;

    // Calculate intermediate values
    u1.ModMulK1(&p2.y, &p1.z);
    v1.ModMulK1(&p2.x, &p1.z);
    u.ModSub(&u1, &p1.y);
    v.ModSub(&v1, &p1.x);

    us2.ModSquareK1(&u);
    vs2.ModSquareK1(&v);
    vs3.ModMulK1(&vs2, &v);

    us2w.ModMulK1(&us2, &p1.z);
    vs2v2.ModMulK1(&vs2, &p1.x);
    
    // Combine `vs2v2` to avoid redundant calculation
    Int _2vs2v2;
    _2vs2v2.ModAdd(&vs2v2, &vs2v2);
    
    a.ModSub(&us2w, &vs3);
    a.ModSub(&_2vs2v2);

    // Calculate final point coordinates
    r.x.ModMulK1(&v, &a);
    vs3u2.ModMulK1(&vs3, &p1.y);
    
    r.y.ModSub(&vs2v2, &a);
    r.y.ModMulK1(&r.y, &u);
    r.y.ModSub(&vs3u2);
    
    r.z.ModMulK1(&vs3, &p1.z);

    return r;
}

Point Secp256K1::Add(Point &p1, Point &p2) {
    Int A, B, C, D, E, F, G, H;
    Point r;

    // Convert affine to Jacobian
    r.z.ModMulK1(&p1.z, &p2.z);
    A.ModMulK1(&p1.x, &p2.z);
    B.ModMulK1(&p2.x, &p1.z);
    C.ModMulK1(&p1.y, &p2.z);
    D.ModMulK1(&p2.y, &p1.z);

    E.ModSub(&B, &A);
    F.ModSub(&D, &C);
    G.ModSquareK1(&E);
    H.ModSquareK1(&A);
    H.ModAdd(&H, &H);
    H.ModAdd(&H, &H);
    G.ModAdd(&G, &H);
    G.ModSub(&G, &G);

    r.x.ModSub(&G, &A);
    r.y.ModSub(&F, &G);
    r.z.ModMulK1(&G, &F);

    return r;
}
Point Secp256K1::DoubleDirect(Point &p) {
    Int _s, _p, a;
    Point r;
    r.z.SetInt32(1);

    _s.ModMulK1(&p.x, &p.x);
    _p.ModAdd(&_s, &_s);
    _p.ModAdd(&_s);

    a.ModAdd(&p.y, &p.y);
    a.ModInv();
    _s.ModMulK1(&_p, &a);     // s = (3*pow2(p.x))*inverse(2*p.y);

    _p.ModMulK1(&_s, &_s);
    a.ModAdd(&p.x, &p.x);
    a.ModNeg();
    r.x.ModAdd(&a, &_p);    // rx = pow2(s) + neg(2*p.x);

    a.ModSub(&r.x, &p.x);

    _p.ModMulK1(&a, &_s);
    r.y.ModAdd(&_p, &p.y);
    r.y.ModNeg();           // ry = neg(p.y + s*(ret.x+neg(p.x)));  

    return r;
}

Point Secp256K1::Double(Point &p) {
    Int A, B, C, D, E, F, G, H;
    Point r;
    
    // Efficiently compute 3*x^2
    A.ModSquareK1(&p.x);
    B.ModAdd(&A, &A); 
    A.ModAdd(&B, &A);
    
    // Compute intermediate values
    C.ModSquareK1(&p.y);
    D.ModSquareK1(&A);
    E.ModMulK1(&B, &C);
    F.ModMulK1(&A, &D);
    
    r.x.ModSub(&E, &F);
    r.y.ModSub(&C, &F);
    
    return r;
}
Int Secp256K1::GetY(Int x, bool isEven) {
    Int s, p;
    s.ModSquareK1(&x);
    p.ModMulK1(&s, &x);
    p.ModAdd(7);
    p.ModSqrt();

    if (p.IsEven() != isEven) {
        p.ModNeg();
    }
    return p;
}

bool Secp256K1::EC(Point &p) {
  Int _s;
  Int _p;

  _s.ModSquareK1(&p.x);
  _p.ModMulK1(&_s,&p.x);
  _p.ModAdd(7);
  _s.ModMulK1(&p.y,&p.y);
  _s.ModSub(&_p);

  return _s.IsZero(); // ( ((pow2(y) - (pow3(x) + 7)) % P) == 0 );
}
