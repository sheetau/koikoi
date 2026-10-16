#define KK_HD __device__ __forceinline__
// core.h is prepended at runtime after the KK_HD definition.
extern "C" __global__ void kk_init(int *s,const int *seeds,int n,int rounds,int bank){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)init(s+i*SS,seeds[i],0,rounds,bank);
}
extern "C" __global__ void kk_step(int *s,const int *a,int *r,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)step(s+i*SS,a[i],r+i*4);
}
extern "C" __global__ void kk_observe(const int *s,float *o,bool *mask,float *target,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)observe(s+i*SS,o+i*OS,mask+i*AS,target+i*48);
}
extern "C" __global__ void kk_reset(int *s,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int *v=s+i*SS;if(v[PH]==4)init(v,v[SEED],v[EP]+1,v[ROUNDS],v[BANK]);}
}
