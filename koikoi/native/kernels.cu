#define KK_HD __device__ __forceinline__
// core.h is prepended at runtime after the KK_HD definition.
extern "C" __global__ void kk_init(int *s,const int *seeds,int n,int rounds,int bank){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)init(s+i*SS,seeds[i],0,rounds,bank);
}
extern "C" __global__ void kk_step(int *s,const int *a,int *r,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)step(s+i*SS,a[i],r+i*4);
}
extern "C" __global__ void kk_step_frozen(int *s,const int *a,int *r,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)step_frozen(s+i*SS,a[i],r+i*4);
}
extern "C" __global__ void kk_solved(const int *s,int *a,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=solved_action(s+i*SS);
}
extern "C" __global__ void kk_observe(const int *s,float *o,bool *mask,float *target,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)observe(s+i*SS,o+i*OS,mask+i*AS,target+i*48);
}
// One block per environment keeps the 15,000-float output writes coalesced.
extern "C" __global__ void kk_published(const int *s,float *o,bool *mask,int n){
    int i=blockIdx.x;if(i>=n)return;
    __shared__ PublishedContext v;
    if(threadIdx.x==0)published_build(s+i*SS,v);
    __syncthreads();
    for(int j=threadIdx.x;j<PUB_SIZE;j+=blockDim.x)
        o[i*PUB_SIZE+j]=published_value(v,j/PUB_COLS,j%PUB_COLS);
    if(threadIdx.x<AS)mask[i*AS+threadIdx.x]=v.active&&legal(s+i*SS,threadIdx.x);
}
extern "C" __global__ void kk_reset(int *s,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){int *v=s+i*SS;if(v[PH]==4)init(v,v[SEED],v[EP]+1,v[ROUNDS],v[BANK]);}
}
extern "C" __global__ void kk_gae(const float *r,const float *v,const bool *d,const int *p,
        const float *next_v,const int *next_p,float *a,float *ret,int h,int n,float lam){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)gae_column(r,v,d,p,next_v,next_p,a,ret,h,n,lam,i);
}
