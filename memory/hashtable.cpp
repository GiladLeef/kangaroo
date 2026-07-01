#include "hashtable.h"
#include "workfile.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <algorithm>
#include <iomanip>
#include <sstream>

namespace {

bool ReadBucketHeader(FILE* f, uint32_t& nbItem, uint32_t& maxItem) {
    return workfile::readValue(f, nbItem) && workfile::readValue(f, maxItem);
}

bool WriteBucketHeader(FILE* f, uint32_t nbItem, uint32_t maxItem) {
    return workfile::writeValue(f, nbItem) && workfile::writeValue(f, maxItem);
}

bool ReadEntry(FILE* f, ENTRY& entry) {
    return workfile::readExact(f, &entry.x, 16) &&
           workfile::readExact(f, &entry.d, 16);
}

bool WriteEntry(FILE* f, const ENTRY& entry) {
    return workfile::writeExact(f, &entry.x, 16) &&
           workfile::writeExact(f, &entry.d, 16);
}

}  // namespace

HashTable::HashTable() {
}

void HashTable::Reset() {
    for(uint32_t h = 0; h < HASH_SIZE; h++) {
        E[h].nbItem = 0;
        std::vector<ENTRY>().swap(E[h].items);
    }
}

uint64_t HashTable::GetNbItem() {
    uint64_t totalItem = 0;
    for(uint64_t h = 0; h < HASH_SIZE; h++) 
        totalItem += (uint64_t)E[h].nbItem;
    return totalItem;
}

uint32_t HashTable::GetBucketSize(uint32_t h) const {
    return E[h].nbItem;
}

const ENTRY* HashTable::GetEntry(uint32_t h,uint32_t i) const {
    return &E[h].items[i];
}

ENTRY HashTable::CreateEntry(int256_t *x,int256_t *d) {

  ENTRY e;
  e.x.i64[0] = x->i64[0];
  e.x.i64[1] = x->i64[1];
  e.x.i64[2] = x->i64[2];
  e.x.i64[3] = x->i64[3];
  e.d.i64[0] = d->i64[0];
  e.d.i64[1] = d->i64[1];
  e.d.i64[2] = d->i64[2];
  e.d.i64[3] = d->i64[3];
  return e;

}

void HashTable::Convert(Int *x,Int *d,uint32_t type,uint64_t *h,int256_t *X,int256_t *D) {
    uint64_t sign = 0;
    uint64_t type64 = (uint64_t)type << 62;
    X->i64[0] = x->bits64[0];
    X->i64[1] = x->bits64[1];
    X->i64[2] = x->bits64[2];
    X->i64[3] = x->bits64[3];
    if(d->bits64[3] > 0x7FFFFFFFFFFFFFFFULL) {
        Int N(d);
        N.ModNegK1order();
        D->i64[0] = N.bits64[0];
        D->i64[1] = N.bits64[1];
        D->i64[2] = N.bits64[2];
        D->i64[3] = N.bits64[3] & 0x3FFFFFFFFFFFFFFFULL;
        sign = 1ULL << 63;
    } else {
        D->i64[0] = d->bits64[0];
        D->i64[1] = d->bits64[1];
        D->i64[2] = d->bits64[2];
        D->i64[3] = d->bits64[3] & 0x3FFFFFFFFFFFFFFFULL;
    }
    D->i64[1] |= sign;
    D->i64[1] |= type64;
    *h = (x->bits64[2] & HASH_MASK);
}
int HashTable::MergeH(uint32_t h, FILE* f1, FILE* f2, FILE* fd, uint32_t* nbDP, uint32_t* duplicate, Int* d1, uint32_t* k1, Int* d2, uint32_t* k2) {
    uint32_t nb1, m1, nb2, m2;
    *duplicate = 0;
    *nbDP = 0;
    ReadBucketHeader(f1, nb1, m1);
    ReadBucketHeader(f2, nb2, m2);

    uint32_t nbd = 0;
    uint32_t md = nb1 + nb2;
    if (md == 0) {
        WriteBucketHeader(fd, md, md);
        return ADD_OK;
    }

    std::vector<ENTRY> output(md);
    ENTRY e1, e2;
    bool end1 = (nb1 == 0), end2 = (nb2 == 0);
    bool collisionFound = false;
    auto drainRemaining = [&](FILE* f, uint32_t remaining) {
        while (remaining > 0) {
            uint32_t batchSize = std::min(remaining, (uint32_t)1024);
            workfile::readExact(f, &output[nbd], sizeof(ENTRY), batchSize);
            nbd += batchSize;
            remaining -= batchSize;
        }
    };
    
    // Read the first entries if available
    if (!end1) ReadEntry(f1, e1);
    if (!end2) ReadEntry(f2, e2);

    // Merge the two sorted arrays in a single pass
    while (!(end1 && end2)) {
        if (!end1 && !end2) {
            int comp = compare(&e1.x, &e2.x);
            if (comp < 0) {
                output[nbd++] = e1;
                end1 = (--nb1 == 0);
                if (!end1) ReadEntry(f1, e1);
            } else if (comp == 0) {
                if((e1.d.i64[0] == e2.d.i64[0]) && (e1.d.i64[1] == e2.d.i64[1]) && 
                   (e1.d.i64[2] == e2.d.i64[2]) && (e1.d.i64[3] == e2.d.i64[3])) {
                    ++(*duplicate);
                } else {
                    CalcDistAndType(e1.d, d1, k1);
                    CalcDistAndType(e2.d, d2, k2);
                    collisionFound = true;
                }
                output[nbd++] = e1;
                
                end1 = (--nb1 == 0);
                end2 = (--nb2 == 0);
                
                if (!end1) ReadEntry(f1, e1);
                if (!end2) ReadEntry(f2, e2);
            } else {
                output[nbd++] = e2;
                end2 = (--nb2 == 0);
                if (!end2) ReadEntry(f2, e2);
            }
        } else if (!end1) {
            drainRemaining(f1, nb1);
            break;
        } else if (!end2) {
            drainRemaining(f2, nb2);
            break;
        }
    }

    md = (nbd + 3) / 4 * 4;
    WriteBucketHeader(fd, nbd, md);
    workfile::writeExact(fd, output.data(), sizeof(ENTRY), nbd);
    *nbDP = nbd;

    return collisionFound ? ADD_COLLISION : ADD_OK;
}

int HashTable::Add(Int *x, Int *d, uint32_t type) {
    int256_t X;
    int256_t D;
    uint64_t h;
    Convert(x,d,type,&h,&X,&D);
    ENTRY e = CreateEntry(&X,&D);
    return Add(h,e);
}

int HashTable::Add(uint64_t h, int256_t *x, int256_t *d) {
    ENTRY e = CreateEntry(x,d);
    return Add(h,e);
}

void HashTable::CalcDistAndType(int256_t d, Int* kDist, uint32_t* kType) {
    *kType = (d.i64[1] & 0x4000000000000000ULL) != 0;
    int sign = (d.i64[1] & 0x8000000000000000ULL) != 0;
    d.i64[1] &= 0x3FFFFFFFFFFFFFFFULL;
    kDist->SetInt32(0);
    kDist->bits64[0] = d.i64[0];
    kDist->bits64[1] = d.i64[1];
    kDist->bits64[2] = d.i64[2];
    kDist->bits64[3] = d.i64[3];
    if(sign) kDist->ModNegK1order();
}
int HashTable::Add(uint64_t h, const ENTRY& e) {
    auto& bucket = E[h].items;
    auto it = std::lower_bound(bucket.begin(), bucket.end(), e, [](const ENTRY& lhs, const ENTRY& rhs) {
        return compare(&lhs.x, &rhs.x) < 0;
    });

    if (it != bucket.end() && compare(&e.x, &it->x) == 0) {
        if((e.d.i64[0] == it->d.i64[0]) && (e.d.i64[1] == it->d.i64[1]) &&
           (e.d.i64[2] == it->d.i64[2]) && (e.d.i64[3] == it->d.i64[3])) {
            return ADD_DUPLICATE;
        }

        CalcDistAndType(it->d, &kDist, &kType);
        return ADD_COLLISION;
    }

    bucket.insert(it, e);
    E[h].nbItem = (uint32_t)bucket.size();
    
    return ADD_OK;
}

int HashTable::compare(const int256_t *i1, const int256_t *i2) {
    const uint64_t *a = i1->i64;
    const uint64_t *b = i2->i64;
    if(a[1] == b[1]) {
        if(a[0] == b[0]) {
            return 0;
        } else {
            return (a[0] > b[0]) ? 1 : -1;
        }
    } else {
        return (a[1] > b[1]) ? 1 : -1;
    }
}

std::string HashTable::GetSizeInfo() {
    uint64_t totalByte = sizeof(E);
    uint64_t usedByte = HASH_SIZE*2*sizeof(uint32_t);
    for (int h = 0; h < HASH_SIZE; h++) {
        totalByte += sizeof(ENTRY) * E[h].items.capacity();
        usedByte += sizeof(ENTRY) * E[h].nbItem;
    }
    const char *unit = "MB";
    double totalMB = (double)totalByte / (1024.0*1024.0);
    double usedMB = (double)usedByte / (1024.0*1024.0);
    if(totalMB > 1024) {
        totalMB /= 1024;
        usedMB /= 1024;
        unit = "GB";
    }
    if(totalMB > 1024) {
        totalMB /= 1024;
        usedMB /= 1024;
        unit = "TB";
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << usedMB << "/" << totalMB << unit;
    return out.str();
}

std::string HashTable::GetStr(int256_t *i) {
    std::ostringstream out;
    out << std::uppercase << std::hex << std::setfill('0');
    for(int n = 3; n >= 0; n--) {
        out << std::setw(8) << i->i32[n];
    }
    return out.str();
}

void HashTable::SaveTable(FILE* f) {
    SaveTable(f,0,HASH_SIZE,true);
}

void HashTable::SaveTable(FILE* f, uint32_t from, uint32_t to, bool printPoint) {
    uint64_t point = GetNbItem() / 16;
    uint64_t pointPrint = 0;
    for(uint32_t h = from; h < to; h++) {
        uint32_t nbItem = E[h].nbItem;
        uint32_t maxItem = (uint32_t)E[h].items.capacity();
        WriteBucketHeader(f, nbItem, maxItem);
        for(uint32_t i = 0; i < nbItem; i++) {
            WriteEntry(f, E[h].items[i]);
            if(printPoint) {
                pointPrint++;
                if(pointPrint > point) {
                    ::printf(".");
                    pointPrint = 0;
                }
            }
        }
    }
}

void HashTable::SeekNbItem(FILE* f, bool restorePos) {
    Reset();
    uint64_t org = (uint64_t)ftello(f);
    SeekNbItem(f,0,HASH_SIZE);
    if( restorePos ) {
        fseeko(f,org,SEEK_SET);
    }
}

void HashTable::SeekNbItem(FILE* f, uint32_t from, uint32_t to) {
    for(uint32_t h = from; h < to; h++) {
        uint32_t nbItem;
        uint32_t maxItem;
        ReadBucketHeader(f, nbItem, maxItem);
        E[h].nbItem = nbItem;
        uint64_t hSize = 32ULL * nbItem;
        fseeko(f,hSize,SEEK_CUR);
    }
}

void HashTable::LoadTable(FILE* f, uint32_t from, uint32_t to) {
    Reset();
    for(uint32_t h = from; h < to; h++) {
        uint32_t nbItem;
        uint32_t maxItem;
        ReadBucketHeader(f, nbItem, maxItem);
        E[h].nbItem = nbItem;
        E[h].items.clear();
        E[h].items.reserve(maxItem);
        E[h].items.resize(nbItem);
        for(uint32_t i = 0; i < nbItem; i++) {
            ReadEntry(f, E[h].items[i]);
        }
    }
}

void HashTable::LoadTable(FILE *f) {
    LoadTable(f,0,HASH_SIZE);
}

void HashTable::PrintInfo() {
    uint16_t max = 0;
    uint32_t maxH = 0;
    uint16_t min = 65535;
    uint32_t minH = 0;
    double std = 0;
    double avg = (double)GetNbItem() / (double)HASH_SIZE;
    for(uint32_t h = 0; h < HASH_SIZE; h++) {
        uint32_t nbItem = E[h].nbItem;
        if(nbItem > max) {
            max = nbItem;
            maxH = h;
        }
        if(nbItem < min) {
            min = nbItem;
            minH = h;
        }
        std += (avg - (double)nbItem)*(avg - (double)nbItem);
    }
    std /= (double)HASH_SIZE;
    std = sqrt(std);
    uint64_t count = GetNbItem();
    ::printf("DP Size   : %s\n",GetSizeInfo().c_str());
    ::printf("DP Count  : %" PRId64 " 2^%.3f\n",count,log2(count));
    ::printf("HT Max    : %d [@ %06X]\n",max,maxH);
    ::printf("HT Min    : %d [@ %06X]\n",min,minH);
    ::printf("HT Avg    : %.2f \n",avg);
    ::printf("HT SDev   : %.2f \n",std);
}
