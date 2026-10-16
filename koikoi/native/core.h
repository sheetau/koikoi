// Allocation-free ABI v1. Compiled unchanged for CPU and GPU.
#ifndef KK_HD
#define KK_HD inline
#endif
enum { SS=208, OS=560, AS=50, P=96, D=97, T=98, PH=99, PEND=100,
       STOCK=101, BEFORE=102, ROUND=103, ROUNDS=104, BANK=105, PT=106,
       KOIS=108, SEED=110, EP=111, HIST=128 };
KK_HD unsigned mix32(unsigned x) {
    x = (x^(x>>16))*0x7feb352du; x = (x^(x>>15))*0x846ca68bu; return x^(x>>16);
}
KK_HD unsigned long long mix64(unsigned long long x) {
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);
}
KK_HD unsigned long long rng_next(unsigned long long &x) { x+=0x9e3779b97f4a7c15ull;return mix64(x); }
KK_HD unsigned below(unsigned long long &x, unsigned n) {
    unsigned long long r, threshold = (0ull-n)%n; do {r=rng_next(x);} while(r<threshold); return (unsigned)(r%n);
}
KK_HD bool has(unsigned long long b, int c) { return (b>>c)&1ull; }
KK_HD int count_bits(unsigned long long b) { int n=0; while(b){b&=b-1; ++n;} return n; }
KK_HD int score_bits(unsigned long long b, int k) {
    const unsigned long long light=(1ull<<0)|(1ull<<8)|(1ull<<28)|(1ull<<40)|(1ull<<44);
    const unsigned long long tane=(1ull<<4)|(1ull<<12)|(1ull<<16)|(1ull<<20)|(1ull<<24)|(1ull<<29)|(1ull<<32)|(1ull<<36)|(1ull<<41);
    const unsigned long long rib=(1ull<<1)|(1ull<<5)|(1ull<<9)|(1ull<<13)|(1ull<<17)|(1ull<<21)|(1ull<<25)|(1ull<<33)|(1ull<<37)|(1ull<<42);
    const unsigned long long red=(1ull<<1)|(1ull<<5)|(1ull<<9), blue=(1ull<<21)|(1ull<<33)|(1ull<<37);
    const unsigned long long kasu=(((1ull<<48)-1)&~(light|tane|rib))|(1ull<<32);
    int n=count_bits(b&light), v=0;
    if(n==5)v=10; else if(n==4)v=has(b,40)?7:8; else if(n==3&&!has(b,40))v=5;
    if(has(b,20)&&has(b,24)&&has(b,36))v+=5;
    if(has(b,8)&&has(b,32))v+=k?3:1;
    if(has(b,28)&&has(b,32))v+=k?3:1;
    n=count_bits(b&tane); if(n>=5)v+=n-4;
    if((b&red)==red)v+=5; if((b&blue)==blue)v+=5;
    if((b&(red|blue))==(red|blue))v+=10;
    n=count_bits(b&rib); if(n>=5)v+=n-4;
    n=count_bits(b&kasu); if(n>=10)v+=n-9;
    return k<=3?v+k:v*(k-2);
}
KK_HD int score_player(const int *s,int p) {
    unsigned long long b=0; for(int c=0;c<48;++c)if(s[c]==3+p)b|=1ull<<c;
    return score_bits(b,s[KOIS+p]);
}
KK_HD bool valid_deck(const int *deck) {
    for(int a=0;a<24;a+=8)for(int m=0;m<12;++m){
        int n=0; for(int j=a;j<a+8;++j)n+=deck[j]/4==m; if(n==4)return false;
    } return true;
}
KK_HD void load_deal(int *s) {
    for(int c=0;c<48;++c)s[c]=-1;
    for(int j=0;j<24;++j)s[s[48+j]]=j/8;
    s[P]=s[D]; s[T]=0; s[PH]=0; s[PEND]=-1; s[STOCK]=47; s[BEFORE]=0;
    s[KOIS]=s[KOIS+1]=0; for(int i=HIST;i<SS;++i)s[i]=0;
}
KK_HD void deal(int *s) {
    unsigned long long rng=mix64((unsigned)s[SEED]^mix64((unsigned)s[EP]+1ull)^mix64((unsigned)s[ROUND]+17ull));
    do {
        for(int i=0;i<48;++i)s[48+i]=i;
        for(int i=47;i>0;--i){int j=below(rng,i+1), tmp=s[48+i];s[48+i]=s[48+j];s[48+j]=tmp;}
    }while(!valid_deck(s+48));
    load_deal(s);
}
KK_HD void init(int *s,int seed,int ep,int rounds,int bankrupt) {
    for(int i=0;i<SS;++i)s[i]=0;
    s[SEED]=seed;s[EP]=ep;s[ROUNDS]=rounds;s[BANK]=bankrupt;s[PT]=s[PT+1]=30;
    s[D]=mix32((unsigned)seed^mix32((unsigned)ep+1))&1u; deal(s);
}
KK_HD int matches(const int *s) {
    int n=0; for(int c=0;c<48;++c)n+=s[c]==2&&c/4==s[PEND]/4;return n;
}
KK_HD bool legal(const int *s,int a) {
    if(a<0||a>=AS)return false;
    if(s[PH]==0)return a<48&&s[a]==s[P];
    if(s[PH]==1||s[PH]==2)return a<48&&s[a]==2&&a/4==s[PEND]/4;
    return s[PH]==3&&a>=48;
}
KK_HD void capture(int *s,int chosen=-1) {
    int n=matches(s), pending=s[PEND];
    if(!n)s[pending]=2;
    else {
        for(int c=0;c<48;++c)if(s[c]==2&&c/4==pending/4&&(n!=2||c==chosen))s[c]=3+s[P];
        s[pending]=3+s[P];
    } s[PEND]=-1;
}
KK_HD void finish(int *s,int winner,int points,int *r) {
    int delta=winner==0?points:-points;
    s[PT]+=delta;s[PT+1]-=delta;r[1]=1;r[3]=delta;
    if(s[ROUND]+1==s[ROUNDS]||(s[BANK]&&(s[PT]<=0||s[PT+1]<=0))){
        s[PH]=4;r[0]=(s[PT]>s[PT+1])-(s[PT]<s[PT+1]);r[2]=1;
    }else{s[ROUND]++;s[D]=winner;deal(s);}
}
KK_HD void end_turn(int *s,int *r) {
    if(s[T]==15)finish(s,s[D],1,r);else{s[T]++;s[P]=1-s[P];s[PH]=0;}
}
KK_HD void after_draw(int *s,int *r) {
    int points=score_player(s,s[P]);
    if(points>s[BEFORE]){
        if(s[T]>=14){s[HIST+s[T]*5+4]=1;finish(s,s[P],points,r);}else s[PH]=3;
    }else end_turn(s,r);
}
KK_HD void draw(int *s,int *r) {
    int c=s[48+s[STOCK]--];s[PEND]=c;s[c]=5;s[HIST+s[T]*5+2]=c+1;
    if(matches(s)==2)s[PH]=2;else{capture(s);after_draw(s,r);}
}
KK_HD void step(int *s,int action,int *r) {
    for(int i=0;i<4;++i)r[i]=0;
    if(!legal(s,action)){r[0]=99;return;}
    int phase=s[PH],h=HIST+s[T]*5;
    if(phase==0){
        s[BEFORE]=score_player(s,s[P]);s[PEND]=action;s[action]=5;s[h]=action+1;
        if(matches(s)==2)s[PH]=1;else{capture(s);draw(s,r);}
    }else if(phase==1||phase==2){
        s[h+(phase==1?1:3)]=action+1;capture(s,action);
        if(phase==1)draw(s,r);else after_draw(s,r);
    }else{
        s[h+4]=action==48?1:2;
        if(action==48)finish(s,s[P],score_player(s,s[P]),r);
        else{s[KOIS+s[P]]++;end_turn(s,r);}
    }
}
KK_HD int solved_action(const int *s) {
    if(s[PH]!=3)return -1;
    int p=s[P],points=score_player(s,p);
    int own=s[PT+p]+points,other=s[PT+1-p]-points;
    if(s[BANK]&&other<=0)return 48;
    if(s[ROUND]+1==s[ROUNDS]){
        if(own>other)return 48;
        if(own<other)return 49;
    }
    return -1;
}
KK_HD void step_frozen(int *s,int action,int *r) {
    if(s[PH]==4){for(int j=0;j<4;++j)r[j]=0;return;}
    step(s,action,r);
}
KK_HD void observe(const int *s,float *o,bool *mask,float *target) {
    for(int i=0;i<OS;++i)o[i]=0;
    int p=s[P],oppn=0;
    for(int c=0;c<48;++c){
        float *f=o+c*11;int loc=s[c];
        f[0]=loc==p;f[1]=loc==2;f[2]=loc==3+p;f[3]=loc==4-p;
        f[4]=loc==1-p||loc==-1;f[5]=loc==5;
        target[c]=loc==1-p;oppn+=loc==1-p;
    }
    for(int j=16;j<24;++j)o[s[48+j]*11+6]=1;
    for(int t=0;t<16;++t){
        int h=HIST+t*5;
        const int offsets[4]={0,2,1,3};
        for(int j=0;j<4;++j){int c=s[h+offsets[j]]-1;if(c>=0)o[c*11+7+j]=(t+1)/16.f;}
        o[528+16+t]=s[h+4]/2.f;
    }
    float *g=o+528;
    if(s[PH]<4)g[s[PH]]=1;
    g[4]=s[T]/16.f;g[5]=s[D]==p;g[6]=s[ROUND]/12.f;g[7]=s[ROUNDS]/12.f;
    g[8]=s[PT+p]/64.f;g[9]=s[PT+1-p]/64.f;g[10]=s[KOIS+p]/8.f;g[11]=s[KOIS+1-p]/8.f;
    g[12]=(s[STOCK]-23)/24.f;g[13]=oppn/8.f;g[14]=s[BEFORE]/32.f;g[15]=(float)s[BANK];
    for(int a=0;a<AS;++a)mask[a]=legal(s,a);
}
