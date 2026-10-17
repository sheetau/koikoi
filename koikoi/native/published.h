// Pinned KoiKoi-AI input layout (89d7895); see THIRD_PARTY_NOTICES.md.
// Only acting-player own cards and public context/history enter this builder.
enum { PUB_ROWS=300, PUB_COLS=50, PUB_SIZE=PUB_ROWS*PUB_COLS };
struct PublishedContext {
    float broadcast[137];
    unsigned long long bits[163];
    int active, offset;
};
KK_HD void published_triple(float *out, double value) {
    double a=value<0?-value:value, sign=(value>0)-(value<0);
    out[0]=(float)(sqrt(a)*sign);
    out[1]=(float)(a*sign*0.5);
    out[2]=(float)(pow(a,1.5)*sign*0.1);
}
KK_HD void published_build(const int *s, PublishedContext &v) {
    for(int i=0;i<137;++i)v.broadcast[i]=0;
    for(int i=0;i<163;++i)v.bits[i]=0;
    v.active=s[ROUNDS]==8&&s[BANK]&&s[PH]>=0&&s[PH]<4;
    v.offset=s[PH]==3?2:0;
    if(!v.active)return;
    const int p=s[P];
    unsigned long long groups[5]={0,0,0,0,0}, initial_field=0;
    for(int c=0;c<48;++c){
        unsigned long long b=1ull<<c;
        if(s[c]==p)groups[0]|=b;
        if(s[c]==2)groups[1]|=b;
        if(s[c]==3+p)groups[2]|=b;
        if(s[c]==4-p)groups[3]|=b;
        if(s[c]==1-p||s[c]==-1)groups[4]|=b;
    }
    for(int i=64;i<72;++i)initial_field|=1ull<<s[i];
    published_triple(v.broadcast+17,(s[PT+p]-s[PT+1-p])/2.0);
    published_triple(v.broadcast+20,score_bits(groups[2],s[KOIS+p]));
    published_triple(v.broadcast+23,score_bits(groups[3],s[KOIS+1-p]));
    v.broadcast[26+s[ROUND]]=1;
    v.broadcast[34+s[T]]=1;
    v.broadcast[50+s[D]]=1;
    v.broadcast[52]=(float)s[KOIS+p];
    v.broadcast[53]=(float)(s[KOIS+p]*s[KOIS+p]);
    v.broadcast[54]=(float)s[KOIS+1-p];
    v.broadcast[55]=(float)(s[KOIS+1-p]*s[KOIS+1-p]);
    for(int t=0;t<16;++t)if(s[HIST+t*5+4]==2)
        v.broadcast[56+(((s[D]+t)%2==p)?0:8)+t/2]=1;
    const unsigned long long light=(1ull<<0)|(1ull<<8)|(1ull<<28)|(1ull<<40)|(1ull<<44);
    const unsigned long long tane=(1ull<<4)|(1ull<<12)|(1ull<<16)|(1ull<<20)|(1ull<<24)|(1ull<<29)|(1ull<<32)|(1ull<<36)|(1ull<<41);
    const unsigned long long rib=(1ull<<1)|(1ull<<5)|(1ull<<9)|(1ull<<13)|(1ull<<17)|(1ull<<21)|(1ull<<25)|(1ull<<33)|(1ull<<37)|(1ull<<42);
    const unsigned long long red=(1ull<<1)|(1ull<<5)|(1ull<<9),blue=(1ull<<21)|(1ull<<33)|(1ull<<37);
    const unsigned long long kasu=(((1ull<<48)-1)&~(light|tane|rib))|(1ull<<32);
    const unsigned long long keys[13]={1ull<<0,1ull<<8,1ull<<28,1ull<<40,1ull<<44,1ull<<32,
        (1ull<<20)|(1ull<<24)|(1ull<<36),tane,red,blue,red|blue,rib,kasu};
    for(int k=0;k<13;++k){
        v.bits[k]=keys[k];
        for(int g=0;g<5;++g)v.broadcast[72+g*13+k]=(float)count_bits(keys[k]&groups[g]);
    }
    for(int m=0;m<12;++m)v.bits[150-137+m]=15ull<<(4*m);
    // Preserve upstream's current-hand "initial" feature and duplicated own pile.
    v.bits[162-137]=groups[0];v.bits[163-137]=initial_field;v.bits[164-137]=groups[4];
    v.bits[165-137]=groups[0];v.bits[166-137]=groups[2];v.bits[167-137]=groups[1];
    v.bits[168-137]=groups[2];v.bits[169-137]=groups[4];
    if(s[PH]==1||s[PH]==2){
        v.bits[170-137]=1ull<<s[PEND];
        v.bits[171-137]=groups[1]&(15ull<<(4*(s[PEND]/4)));
    }
    // Reconstruct field captures from public history, without either initial hand
    // or the saved stock permutation. Unresolved two-card picks have no capture log.
    unsigned long long board=initial_field;
    for(int t=0;t<=s[T];++t){
        int h=HIST+t*5, first=s[h]-1;
        if(first<0)break;
        int row=172+(s[T]-t)*8-137;
        for(int part=0;part<2;++part){
            int card=s[h+part*2]-1;
            if(card<0)break;
            unsigned long long bit=1ull<<card, paired=board&(15ull<<(4*(card/4)));
            int n=count_bits(paired), chosen=s[h+part*2+1]-1;
            v.bits[row+part*4+(n?0:1)]=bit;
            if(n==2&&chosen<0)break;
            unsigned long long collected=n==2?(1ull<<chosen):paired;
            v.bits[row+part*4+2]=collected;
            v.bits[row+part*4+3]=paired&~collected;
            if(n)board&=~collected;else board|=bit;
        }
    }
}
KK_HD float published_value(const PublishedContext &v,int row,int col) {
    if(!v.active)return 0;
    if(v.offset&&col<2){
        if(row==col)return 1;
        return row<137?v.broadcast[row]:0;
    }
    int card=col-v.offset;
    if(card<0||card>=48)return 0;
    return row<137?v.broadcast[row]:(float)has(v.bits[row-137],card);
}
KK_HD void published_observe(const int *s,float *out,bool *mask) {
    PublishedContext v;published_build(s,v);
    for(int i=0;i<PUB_SIZE;++i)out[i]=published_value(v,i/PUB_COLS,i%PUB_COLS);
    for(int a=0;a<AS;++a)mask[a]=v.active&&legal(s,a);
}
