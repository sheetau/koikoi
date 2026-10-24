// Research ABI: original core remains byte-preserved. Shared by CPU and CUDA.
namespace publicstock {
using namespace kkbase;
enum { FLAGS=112, AUTO_ROUNDS=113, COMPLETED_ROUNDS=114, PUBLIC=115 };
KK_HD int raw(const int *s,int p) {
    unsigned long long b=0;for(int c=0;c<48;++c)if(s[c]==3+p)b|=1ull<<c;
    int f=s[FLAGS],k=(f&128)?0:s[KOIS+p],v=score_bits(b,k);
    if(f&64){int n=(has(b,8)&&has(b,32))+(has(b,28)&&has(b,32));v+=(5-(k?3:1))*n*(k>3?k-2:1);}
    return v;
}
KK_HD int payment(const int *s,int p) {
    int f=s[FLAGS],v=raw(s,p);if(f&16)v+=raw(s,1-p);
    if((f&2)&&v>=7)v*=2;if((f&4)&&s[KOIS+1-p])v*=2;if((f&8)&&s[KOIS+p])v*=2;return v;
}
KK_HD bool valid(const int *deck,int flags) {
    for(int a=(flags&1)?16:0;a<24;a+=8)for(int m=0;m<12;++m){
        int n=0;for(int j=a;j<a+8;++j)n+=deck[j]/4==m;if(n==4)return false;
    }return true;
}
KK_HD bool opening(const int *s,int p) {
    int pairs=0;for(int m=0;m<12;++m){int n=0;for(int c=4*m;c<4*m+4;++c)n+=s[c]==p;
        if(n==4)return true;pairs+=n==2;}return pairs==4;
}
KK_HD void settle_only(int *s,int winner,int points) {
    s[PT+winner]+=points;s[PT+1-winner]-=points;s[COMPLETED_ROUNDS]++;
    if(s[ROUND]+1==s[ROUNDS]||(s[BANK]&&(s[PT]<=0||s[PT+1]<=0)))s[PH]=4;
    else{s[ROUND]++;s[D]=winner;s[PH]=5;}
}
KK_HD void expose(int *s){s[PUBLIC]=s[48+s[STOCK]]+1;s[PUBLIC+1]=s[47+s[STOCK]]+1;}
KK_HD void begin(int *s) {
    do {
        unsigned long long rng=mix64((unsigned)s[SEED]^mix64((unsigned)s[EP]+1ull)^mix64((unsigned)s[ROUND]+17ull));
        do{for(int i=0;i<48;++i)s[48+i]=i;
            for(int i=47;i>0;--i){int j=below(rng,i+1),tmp=s[48+i];s[48+i]=s[48+j];s[48+j]=tmp;}
        }while(!valid(s+48,s[FLAGS]));
        load_deal(s);s[PUBLIC]=s[PUBLIC+1]=0;if(s[FLAGS]&512)expose(s);
        if(s[FLAGS]&1){int p=s[D];if(!opening(s,p))p=1-p;
            if(opening(s,p)){s[AUTO_ROUNDS]++;settle_only(s,p,6);}}
    }while(s[PH]==5);
}
KK_HD void init_variant(int *s,int seed,int ep,int flags,int rounds,int bank) {
    for(int i=0;i<SS;++i)s[i]=0;s[SEED]=seed;s[EP]=ep;s[ROUNDS]=rounds;s[BANK]=bank;
    s[PT]=s[PT+1]=30;s[FLAGS]=flags;s[D]=mix32((unsigned)seed^mix32((unsigned)ep+1))&1u;begin(s);
}
KK_HD void settle(int *s,int p,int points){settle_only(s,p,points);if(s[PH]==5)begin(s);}
KK_HD void next_turn(int *s) {
    if(s[T]==15){bool z=s[FLAGS]&32;settle(s,z?1-s[D]:s[D],z?0:1);}
    else{s[T]++;s[P]=1-s[P];s[PH]=0;}
}
KK_HD void after(int *s) {
    if(raw(s,s[P])>s[BEFORE]){
        if(s[T]>=14||((s[FLAGS]&256)&&(s[KOIS]||s[KOIS+1]))){s[HIST+s[T]*5+4]=1;settle(s,s[P],payment(s,s[P]));}
        else s[PH]=3;
    }else next_turn(s);
}
KK_HD bool stock_legal(const int *s,int a){
    if(s[PH]==5)return a>=0&&a<48&&(a+1==s[PUBLIC]||a+1==s[PUBLIC+1]);
    return legal(s,a);
}
KK_HD void take_stock(int *s){
    s[PUBLIC]=s[PUBLIC+1];s[PUBLIC+1]=0;
    if(s[FLAGS]&512){s[PUBLIC]=s[47+s[STOCK]]+1;s[PUBLIC+1]=s[46+s[STOCK]]+1;}
    int c=s[48+s[STOCK]--];s[PEND]=c;s[c]=5;s[HIST+s[T]*5+2]=c+1;
    if(matches(s)==2)s[PH]=2;else{capture(s);after(s);}}
KK_HD void reveal(int *s){if(s[FLAGS]&1024){expose(s);s[PH]=5;}else take_stock(s);}
KK_HD void step_variant(int *s,int a,int *r) {
    for(int j=0;j<4;++j)r[j]=0;if(s[PH]==4)return;if(!stock_legal(s,a)){r[0]=99;return;}
    int pt=s[PT],rounds=s[COMPLETED_ROUNDS],phase=s[PH],h=HIST+s[T]*5;
    if(phase==5){int top=48+s[STOCK];if(s[top]!=a){
        int tmp=s[top];s[top]=s[top-1];s[top-1]=tmp;
        tmp=s[PUBLIC];s[PUBLIC]=s[PUBLIC+1];s[PUBLIC+1]=tmp;}take_stock(s);}
    else if(phase==0){s[BEFORE]=raw(s,s[P]);s[PEND]=a;s[a]=5;s[h]=a+1;
        if(matches(s)==2)s[PH]=1;else{capture(s);reveal(s);}}
    else if(phase==1||phase==2){s[h+(phase==1?1:3)]=a+1;capture(s,a);if(phase==1)reveal(s);else after(s);}
    else{s[h+4]=a==48?1:2;if(a==48)settle(s,s[P],payment(s,s[P]));else{s[KOIS+s[P]]++;next_turn(s);}}
    r[1]=s[COMPLETED_ROUNDS]-rounds;r[3]=s[PT]-pt;r[2]=s[PH]==4;
    if(r[2])r[0]=(s[PT]>s[PT+1])-(s[PT]<s[PT+1]);
}
KK_HD int solved(const int *s) {
    if(s[PH]!=3)return -1;int p=s[P],v=payment(s,p),own=s[PT+p]+v,other=s[PT+1-p]-v;
    if(s[BANK]&&other<=0)return 48;
    if(s[ROUND]+1==s[ROUNDS]){if(own>other)return 48;if(own<other)return 49;}
    return -1;
}
KK_HD void observation(const int *s,float *o,bool *mask){
    float unused[48];kkbase::observe(s,o,mask,unused);
    for(int j=560;j<660;++j)o[j]=0;
    for(int j=0;j<2;++j){int c=s[PUBLIC+j]-1;if(c>=0){o[560+48*j+c]=1;o[c*11+4]=0;}}
    o[656]=s[PH]==5;o[657]=(s[FLAGS]&512)!=0;o[658]=(s[FLAGS]&1024)!=0;o[659]=(s[FLAGS]&24)!=0;
    for(int a=0;a<50;++a)mask[a]=stock_legal(s,a);
}
}
