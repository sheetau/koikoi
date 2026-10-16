#include "core.h"
#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif
EXPORT void kk_init(int *s,const int *seeds,int n,int rounds,int bank){
    for(int i=0;i<n;++i)init(s+i*SS,seeds[i],0,rounds,bank);
}
EXPORT void kk_step(int *s,const int *a,int *r,int n){
    for(int i=0;i<n;++i)step(s+i*SS,a[i],r+i*4);
}
EXPORT void kk_observe(const int *s,float *o,bool *mask,float *target,int n){
    for(int i=0;i<n;++i)observe(s+i*SS,o+i*OS,mask+i*AS,target+i*48);
}
EXPORT void kk_reset(int *s,int n){
    for(int i=0;i<n;++i){int *v=s+i*SS;if(v[PH]==4)init(v,v[SEED],v[EP]+1,v[ROUNDS],v[BANK]);}
}
EXPORT int kk_score(unsigned long long b,int k){return score_bits(b,k);}
