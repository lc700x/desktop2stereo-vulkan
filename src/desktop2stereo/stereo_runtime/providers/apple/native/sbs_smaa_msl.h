/*
 * SMAA 1x High lookup-driven display antialiasing kernels.
 * Lookup tables and algorithm structure derive from SMAA by Jorge Jimenez et al.
 * See shaders/SMAA_LICENSE.txt. All processing happens after DIBR visibility.
 */
static const char *D2S_SMAA_MSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct SmaaParams {
    uint width, height, channels, layout, eye_width, batches, half_sbs, reserved;
};

static inline uint smaa_src_index(uint x, uint y, uint b, uint c, constant SmaaParams &p) {
    if ((p.reserved & 1u) != 0u) y = p.height - 1u - y;
    if (p.layout == 0u) return ((y * p.width + x) * p.batches + b) * p.channels + c;
    return ((b * p.channels + c) * p.height + y) * p.width + x;
}
static inline uint smaa_dst_index(uint x, uint y, uint b, uint c, constant SmaaParams &p) {
    uint width=p.half_sbs!=0u?p.width/2u:p.width;
    if(p.layout==0u) return ((y*width+x)*p.batches+b)*p.channels+c;
    return ((b*p.channels+c)*p.height+y)*width+x;
}
static inline float smaa_color(device const uchar *u8, device const float *f32,
                               uint x, uint y, uint b, uint c, constant SmaaParams &p) {
    uint i = smaa_src_index(x, y, b, c, p);
    return p.layout == 0u || p.layout == 2u ? float(u8[i]) * (1.0f / 255.0f) : f32[i];
}
static inline float smaa_luma(device const uchar *u8, device const float *f32,
                              int x, int y, uint b, constant SmaaParams &p,
                              uint eye_first, uint eye_last) {
    x = clamp(x, int(eye_first), int(eye_last));
    y = clamp(y, 0, int(p.height) - 1);
    float3 c = float3(smaa_color(u8,f32,uint(x),uint(y),b,0,p),
                      smaa_color(u8,f32,uint(x),uint(y),b,1,p),
                      smaa_color(u8,f32,uint(x),uint(y),b,2,p));
    return dot(c, float3(0.2126f, 0.7152f, 0.0722f));
}
static inline float2 smaa_edge_at(device const uchar *edges, int x, int y, uint b,
                                   constant SmaaParams &p, uint eye_first, uint eye_last) {
    x = clamp(x, int(eye_first), int(eye_last));
    y = clamp(y, 0, int(p.height) - 1);
    uint i = ((b * p.height + uint(y)) * p.width + uint(x)) * 2u;
    return float2(edges[i], edges[i + 1u]) * (1.0f / 255.0f);
}
static inline float2 smaa_edge_sample(device const uchar *edges, float2 uv, uint b,
                                      constant SmaaParams &p, uint eye_first, uint eye_last) {
    float2 pos = uv * float2(p.width, p.height) - 0.5f;
    int2 base = int2(floor(pos));
    float2 f = fract(pos);
    float2 a = smaa_edge_at(edges, base.x, base.y, b, p, eye_first, eye_last);
    float2 c = smaa_edge_at(edges, base.x + 1, base.y, b, p, eye_first, eye_last);
    float2 d = smaa_edge_at(edges, base.x, base.y + 1, b, p, eye_first, eye_last);
    float2 e = smaa_edge_at(edges, base.x + 1, base.y + 1, b, p, eye_first, eye_last);
    return mix(mix(a,c,f.x), mix(d,e,f.x), f.y);
}
static inline float smaa_lut(device const uchar *lut, uint width, uint height, uint channels,
                             float2 uv, uint channel) {
    float2 pos = clamp(uv * float2(width,height) - 0.5f, 0.0f, float2(width-1,height-1));
    uint2 a = uint2(floor(pos)); uint2 z = min(a + 1u, uint2(width-1,height-1));
    float2 f = fract(pos);
    float p00=float(lut[(a.y*width+a.x)*channels+channel]);
    float p10=float(lut[(a.y*width+z.x)*channels+channel]);
    float p01=float(lut[(z.y*width+a.x)*channels+channel]);
    float p11=float(lut[(z.y*width+z.x)*channels+channel]);
    return mix(mix(p00,p10,f.x),mix(p01,p11,f.x),f.y) * (1.0f/255.0f);
}
static inline float smaa_search_length(device const uchar *search, float2 e, float offset) {
    float2 uv = (float2(32.0f,-32.0f)*e + float2(66.0f*offset+0.5f,32.5f)) / float2(64.0f,16.0f);
    return smaa_lut(search,64,16,1,uv,0);
}
static inline float2 smaa_area(device const uchar *area, float2 dist, float e1, float e2, float offset) {
    float2 pos = float2(16.0f) * round(4.0f * float2(e1,e2)) + dist;
    float2 uv = (pos + 0.5f) / float2(160.0f,560.0f);
    uv.y += (1.0f/7.0f) * offset;
    return float2(smaa_lut(area,160,560,2,uv,0),smaa_lut(area,160,560,2,uv,1));
}
static inline float2 smaa_area_diag(device const uchar *area, float2 dist, float2 e, float offset) {
    float2 pos = 20.0f * e + dist;
    float2 uv = (pos + 0.5f) / float2(160.0f,560.0f);
    uv.x += 0.5f; uv.y += (1.0f/7.0f)*offset;
    return float2(smaa_lut(area,160,560,2,uv,0),smaa_lut(area,160,560,2,uv,1));
}
static inline float2 smaa_decode_diag(float2 e) {
    e.x=e.x*abs(5.0f*e.x-3.75f); return round(e);
}
static inline float2 smaa_search_diag1(device const uchar *edges, float2 uv, float2 dir,
                                       uint b, constant SmaaParams &p, uint ef, uint el) {
    float z=-1.0f, w=1.0f;
    for(uint i=0;i<8u;i++) { if(!(z<7.0f && w>0.9f)) break;
        uv+=dir/float2(p.width,p.height); z+=1.0f;
        float2 e=smaa_edge_sample(edges,uv,b,p,ef,el); w=dot(e,float2(0.5f)); }
    return float2(z,w);
}
static inline float2 smaa_search_diag2(device const uchar *edges, float2 uv, float2 dir,
                                       uint b, constant SmaaParams &p, uint ef, uint el) {
    uv.x+=0.25f/float(p.width); float z=-1.0f, w=1.0f;
    for(uint i=0;i<8u;i++) { if(!(z<7.0f && w>0.9f)) break;
        uv+=dir/float2(p.width,p.height); z+=1.0f;
        float2 e=smaa_decode_diag(smaa_edge_sample(edges,uv,b,p,ef,el)); w=dot(e,float2(0.5f)); }
    return float2(z,w);
}
static inline float2 smaa_diag_weights(device const uchar *edges,
                                       device const uchar *area,
                                       float2 uv, float2 e, uint b, constant SmaaParams &p,
                                       uint ef, uint el) {
    float2 weights=0.0f; float2 end;
    float4 d;
    float2 diag1a=(p.reserved&4u)!=0u?float2(-1,-1):float2(-1,1);
    float2 diag1b=(p.reserved&4u)!=0u?float2(1,1):float2(1,-1);
    float2 diag2a=(p.reserved&4u)!=0u?float2(-1,1):float2(-1,-1);
    float2 diag2b=(p.reserved&4u)!=0u?float2(1,-1):float2(1,1);
    if(e.x>0.0f) { float2 q=smaa_search_diag1(edges,uv,diag1a,b,p,ef,el);
        d.xz=q; d.x+=float(q.y>0.9f); }
    else d.xz=0.0f;
    d.yw=smaa_search_diag1(edges,uv,diag1b,b,p,ef,el);
    if(d.x+d.y>2.0f) {
        float2 c0=uv+float2(-d.x+0.25f,d.x)/float2(p.width,p.height);
        float2 c1=uv+float2(d.y,-d.y-0.25f)/float2(p.width,p.height);
        float2 c=smaa_edge_sample(edges,c0+float2(-1.0f/p.width,0),b,p,ef,el);
        float2 z=smaa_edge_sample(edges,c1+float2(1.0f/p.width,0),b,p,ef,el);
        // Match SMAA's c.yxwz swizzle: decode the bilinear values from R/B,
        // then pair the decoded crossing edge with the binary orthogonal edge.
        float2 cross=2.0f*float2(round(c.y),round(z.y))+
                     float2(round(c.x*abs(5.0f*c.x-3.75f)),
                            round(z.x*abs(5.0f*z.x-3.75f)));
        if(d.z>0.9f) cross.x=0.0f; if(d.w>0.9f) cross.y=0.0f;
        weights+=smaa_area_diag(area,d.xy,cross,0.0f);
    }
    d.xz=smaa_search_diag2(edges,uv,diag2a,b,p,ef,el);
    if(smaa_edge_sample(edges,uv+float2(1.0f/p.width,0),b,p,ef,el).x>0.0f) {
        float2 q=smaa_search_diag2(edges,uv,diag2b,b,p,ef,el); d.yw=q; d.y+=float(q.y>0.9f);
    } else d.yw=0.0f;
    if(d.x+d.y>2.0f) {
        float2 c0=uv+float2(-d.x,-d.x)/float2(p.width,p.height);
        float2 c1=uv+float2(d.y,d.y)/float2(p.width,p.height);
        float2 c=smaa_edge_sample(edges,c0+float2(-1.0f/p.width,0),b,p,ef,el);
        float2 z=smaa_edge_sample(edges,c0+float2(0,-1.0f/p.height),b,p,ef,el);
        float2 q=smaa_edge_sample(edges,c1+float2(1.0f/p.width,0),b,p,ef,el);
        float2 cross=2.0f*float2(c.y,q.y)+float2(z.x,q.x);
        if(d.z>0.9f) cross.x=0.0f; if(d.w>0.9f) cross.y=0.0f;
        weights+=smaa_area_diag(area,d.xy,cross,0.0f).yx;
    }
    return weights;
}
static inline float smaa_x_left(device const uchar *edges, device const uchar *search,
                                float2 uv, float end, uint b, constant SmaaParams &p,
                                uint ef, uint el) {
    float2 e=float2(0,1);
    for(uint i=0;i<16u;i++) { if(!(uv.x>end && e.y>0.8281f && e.x==0.0f)) break;
        e=smaa_edge_sample(edges,uv,b,p,ef,el); uv.x-=2.0f/float(p.width); }
    float off=-(255.0f/127.0f)*smaa_search_length(search,e,0.0f)+3.25f;
    return uv.x+off/float(p.width);
}
static inline float smaa_x_right(device const uchar *edges, device const uchar *search,
                                 float2 uv, float end, uint b, constant SmaaParams &p,
                                 uint ef, uint el) {
    float2 e=float2(0,1);
    for(uint i=0;i<16u;i++) { if(!(uv.x<end && e.y>0.8281f && e.x==0.0f)) break;
        e=smaa_edge_sample(edges,uv,b,p,ef,el); uv.x+=2.0f/float(p.width); }
    float off=-(255.0f/127.0f)*smaa_search_length(search,e,0.5f)+3.25f;
    return uv.x-off/float(p.width);
}
static inline float smaa_y_up(device const uchar *edges, device const uchar *search,
                              float2 uv, float end, uint b, constant SmaaParams &p,
                              uint ef, uint el) {
    float2 e=float2(1,0);
    for(uint i=0;i<16u;i++) { if(!(uv.y>end && e.x>0.8281f && e.y==0.0f)) break;
        e=smaa_edge_sample(edges,uv,b,p,ef,el); uv.y-=2.0f/float(p.height); }
    float off=-(255.0f/127.0f)*smaa_search_length(search,e.yx,0.0f)+3.25f;
    return uv.y+off/float(p.height);
}
static inline float smaa_y_down(device const uchar *edges, device const uchar *search,
                                float2 uv, float end, uint b, constant SmaaParams &p,
                                uint ef, uint el) {
    float2 e=float2(1,0);
    for(uint i=0;i<16u;i++) { if(!(uv.y<end && e.x>0.8281f && e.y==0.0f)) break;
        e=smaa_edge_sample(edges,uv,b,p,ef,el); uv.y+=2.0f/float(p.height); }
    float off=-(255.0f/127.0f)*smaa_search_length(search,e.yx,0.5f)+3.25f;
    return uv.y-off/float(p.height);
}
static inline float4 smaa_weights(device const uchar *edges,
                                  device const uchar *area, device const uchar *search,
                                  uint x, uint y, uint b,
                                  constant SmaaParams &p, uint ef, uint el) {
    float2 uv=(float2(x,y)+0.5f)/float2(p.width,p.height);
    float2 e=smaa_edge_at(edges,int(x),int(y),b,p,ef,el); float4 w=0.0f;
    bool diagonal=false;
    if(e.y>0.0f) {
        float2 dw=smaa_diag_weights(edges,area,uv,e,b,p,ef,el);
        if(dw.x+dw.y>1e-5f) { w.xy=dw; diagonal=true; }
    }
    if(e.y>0.0f && !diagonal) {
        float2 a=uv+float2(-0.25f/float(p.width),-0.125f/float(p.height));
        float2 z=uv+float2(1.25f/float(p.width),-0.125f/float(p.height));
        float xl=smaa_x_left(edges,search,a,uv.x-32.0f/float(p.width),b,p,ef,el);
        float xr=smaa_x_right(edges,search,z,uv.x+32.0f/float(p.width),b,p,ef,el);
        float cy=uv.y-0.25f/float(p.height);
        float e1=smaa_edge_sample(edges,float2(xl,cy),b,p,ef,el).x;
        float e2=smaa_edge_sample(edges,float2(xr+1.0f/float(p.width),cy),b,p,ef,el).x;
        float2 d=abs(round(float2((xl-uv.x)*p.width,(xr-uv.x)*p.width)));
        w.xy=smaa_area(area,sqrt(d),e1,e2,0.0f);
        float2 lr=step(d,d.yx); float2 rounding=(1.0f-0.25f)*lr;
        rounding/=max(lr.x+lr.y,1.0f);
        float f0=1.0f-rounding.x*smaa_edge_sample(edges,uv+float2(0,1.0f/p.height),b,p,ef,el).x-rounding.y*smaa_edge_sample(edges,uv+float2(1.0f/p.width,1.0f/p.height),b,p,ef,el).x;
        float f1=1.0f-rounding.x*smaa_edge_sample(edges,uv+float2(0,-2.0f/p.height),b,p,ef,el).x-rounding.y*smaa_edge_sample(edges,uv+float2(1.0f/p.width,-2.0f/p.height),b,p,ef,el).x;
        w.xy*=saturate(float2(f0,f1));
    }
    if(e.x>0.0f && !diagonal) {
        float2 a=uv+float2(-0.125f/float(p.width),-0.25f/float(p.height));
        float2 z=uv+float2(-0.125f/float(p.width),1.25f/float(p.height));
        float yu=smaa_y_up(edges,search,a,uv.y-32.0f/float(p.height),b,p,ef,el);
        float yd=smaa_y_down(edges,search,z,uv.y+32.0f/float(p.height),b,p,ef,el);
        float cx=uv.x-0.25f/float(p.width);
        float e1=smaa_edge_sample(edges,float2(cx,yu),b,p,ef,el).y;
        float e2=smaa_edge_sample(edges,float2(cx,yd+1.0f/float(p.height)),b,p,ef,el).y;
        float2 d=abs(round(float2((yu-uv.y)*p.height,(yd-uv.y)*p.height)));
        w.zw=smaa_area(area,sqrt(d),e1,e2,0.0f);
        float2 lr=step(d,d.yx); float2 rounding=0.75f*lr;
        rounding/=max(lr.x+lr.y,1.0f);
        float f0=1.0f-rounding.x*smaa_edge_sample(edges,uv+float2(1.0f/p.width,0),b,p,ef,el).y-rounding.y*smaa_edge_sample(edges,uv+float2(1.0f/p.width,1.0f/p.height),b,p,ef,el).y;
        float f1=1.0f-rounding.x*smaa_edge_sample(edges,uv+float2(-2.0f/p.width,0),b,p,ef,el).y-rounding.y*smaa_edge_sample(edges,uv+float2(-2.0f/p.width,1.0f/p.height),b,p,ef,el).y;
        w.zw*=saturate(float2(f0,f1));
    }
    return w;
}
static inline float3 smaa_decode(float3 c) {
    return select(c/12.92f,pow(max((c+0.055f)/1.055f,0.0f),2.4f),c>0.04045f);
}
static inline float3 smaa_encode(float3 c) {
    c=clamp(c,0.0f,1.0f);
    return select(c*12.92f,1.055f*pow(c,1.0f/2.4f)-0.055f,c>0.0031308f);
}
static inline float smaa_weight_at(device const uchar *weights, uint i) {
    return float(weights[i]) * (1.0f/255.0f);
}
static inline float4 smaa_resolve_at(device const uchar *src_u8, device const float *src_f32,
                                     device const uchar *weights, uint x, uint y, uint b,
                                     constant SmaaParams &p, uint ef, uint el) {
    uint i=((b*p.height+y)*p.width+x)*4u;
    float4 a;
    a.x=smaa_weight_at(weights,((b*p.height+y)*p.width+min(x+1u,el))*4u+3u);
    a.y=smaa_weight_at(weights,((b*p.height+min(y+1u,p.height-1u))*p.width+x)*4u+1u);
    a.z=smaa_weight_at(weights,i+2u); a.w=smaa_weight_at(weights,i);
    if(dot(a,float4(1.0f))<1e-5f) {
        return float4(smaa_color(src_u8,src_f32,x,y,b,0,p),smaa_color(src_u8,src_f32,x,y,b,1,p),smaa_color(src_u8,src_f32,x,y,b,2,p),p.channels==4u?smaa_color(src_u8,src_f32,x,y,b,3,p):1.0f);
    }
    bool h=max(a.x,a.z)>max(a.y,a.w);
    float2 blend=h?float2(a.x,a.z):float2(a.y,a.w);
    blend/=max(blend.x+blend.y,1e-6f);
    uint right=uint(clamp(int(x)+1,int(ef),int(el))), left=uint(clamp(int(x)-1,int(ef),int(el)));
    uint down=min(y+1u,p.height-1u), up=y>0u?y-1u:0u;
    float3 c=float3(smaa_color(src_u8,src_f32,x,y,b,0,p),smaa_color(src_u8,src_f32,x,y,b,1,p),smaa_color(src_u8,src_f32,x,y,b,2,p));
    float3 c0, c1;
    if(h) {
        float3 cr=float3(smaa_color(src_u8,src_f32,right,y,b,0,p),smaa_color(src_u8,src_f32,right,y,b,1,p),smaa_color(src_u8,src_f32,right,y,b,2,p));
        float3 cl=float3(smaa_color(src_u8,src_f32,left,y,b,0,p),smaa_color(src_u8,src_f32,left,y,b,1,p),smaa_color(src_u8,src_f32,left,y,b,2,p));
        c0=mix(smaa_decode(c),smaa_decode(cr),a.x); c1=mix(smaa_decode(c),smaa_decode(cl),a.z);
    } else {
        float3 cd=float3(smaa_color(src_u8,src_f32,x,down,b,0,p),smaa_color(src_u8,src_f32,x,down,b,1,p),smaa_color(src_u8,src_f32,x,down,b,2,p));
        float3 cu=float3(smaa_color(src_u8,src_f32,x,up,b,0,p),smaa_color(src_u8,src_f32,x,up,b,1,p),smaa_color(src_u8,src_f32,x,up,b,2,p));
        c0=mix(smaa_decode(c),smaa_decode(cd),a.y); c1=mix(smaa_decode(c),smaa_decode(cu),a.w);
    }
    float4 out=float4(smaa_encode(c0*blend.x+c1*blend.y),1.0f);
    if(p.channels==4u) out.a=smaa_color(src_u8,src_f32,x,y,b,3,p);
    return out;
}

kernel void d2s_smaa_edges(device const uchar *src_u8 [[buffer(0)]], device const float *src_f32 [[buffer(1)]],
                           device uchar *edge_output [[buffer(2)]], constant SmaaParams &p [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
    uint x=gid.x,y=gid.y,b=gid.z; if(x>=p.width||y>=p.height||b>=p.batches)return;
    if((p.reserved&1u) != 0u) y=p.height-1u-y;
    uint eye=p.eye_width==0u?p.width:p.eye_width; uint ef=(x/eye)*eye, el=min(ef+eye,p.width)-1u;
    float m=smaa_luma(src_u8,src_f32,int(x),int(y),b,p,ef,el);
    float l=smaa_luma(src_u8,src_f32,int(x)-1,int(y),b,p,ef,el);
    int top_y=(p.reserved&2u)!=0u?int(y)+1:int(y)-1;
    float t=smaa_luma(src_u8,src_f32,int(x),top_y,b,p,ef,el);
    float r=smaa_luma(src_u8,src_f32,int(x)+1,int(y),b,p,ef,el), d=smaa_luma(src_u8,src_f32,int(x),int(y)+1,b,p,ef,el);
    float ll=smaa_luma(src_u8,src_f32,int(x)-2,int(y),b,p,ef,el), tt=smaa_luma(src_u8,src_f32,int(x),int(y)-2,b,p,ef,el);
    float2 delta=abs(m-float2(l,t)); float2 flags=step(float2(0.1f),delta);
    float2 maxDelta=max(delta,abs(m-float2(r,d)));
    maxDelta=max(maxDelta,abs(float2(l,t)-float2(ll,tt)));
    flags*=step(maxDelta,float2(2.0f)*delta);
    uint i=((b*p.height+y)*p.width+x)*2u;
    flags=round(flags*255.0f);
    edge_output[i]=uchar(flags.x); edge_output[i+1u]=uchar(flags.y);
}

kernel void d2s_smaa_weights(device const uchar *edges [[buffer(0)]], device const uchar *area [[buffer(1)]],
                             device const uchar *search [[buffer(2)]], device uchar *weights [[buffer(3)]],
                             constant SmaaParams &p [[buffer(4)]], uint3 gid [[thread_position_in_grid]]) {
    uint x=gid.x,y=gid.y,b=gid.z; if(x>=p.width||y>=p.height||b>=p.batches)return;
    if((p.reserved&1u) != 0u) y=p.height-1u-y;
    uint eye=p.eye_width==0u?p.width:p.eye_width; uint ef=(x/eye)*eye, el=min(ef+eye,p.width)-1u;
    float4 w=smaa_weights(edges,area,search,x,y,b,p,ef,el);
    uint i=((b*p.height+y)*p.width+x)*4u;
    weights[i]=uchar(clamp(floor(w.x*255.0f+0.5f),0.0f,255.0f));
    weights[i+1u]=uchar(clamp(floor(w.y*255.0f+0.5f),0.0f,255.0f));
    weights[i+2u]=uchar(clamp(floor(w.z*255.0f+0.5f),0.0f,255.0f));
    weights[i+3u]=uchar(clamp(floor(w.w*255.0f+0.5f),0.0f,255.0f));
}

kernel void d2s_smaa_debug_diag(device const uchar *edges [[buffer(0)]], device const uchar *area [[buffer(1)]],
                                device float *debug [[buffer(2)]], constant SmaaParams &p [[buffer(3)]],
                                uint3 gid [[thread_position_in_grid]]) {
    uint x=gid.x,y=gid.y,b=gid.z; if(x>=p.width||y>=p.height||b>=p.batches)return;
    if((p.reserved&1u) != 0u) y=p.height-1u-y;
    uint eye=p.eye_width==0u?p.width:p.eye_width; uint ef=(x/eye)*eye, el=min(ef+eye,p.width)-1u;
    float2 uv=(float2(x,y)+0.5f)/float2(p.width,p.height);
    float2 e=smaa_edge_at(edges,int(x),int(y),b,p,ef,el);
    float2 dir1=(p.reserved&4u)!=0u?float2(1,1):float2(1,-1);
    float2 dir2=(p.reserved&4u)!=0u?float2(1,-1):float2(1,1);
    float2 q1=smaa_search_diag1(edges,uv,dir1,b,p,ef,el);
    float2 q2=smaa_search_diag2(edges,uv,dir2,b,p,ef,el);
    float2 dw=smaa_diag_weights(edges,area,uv,e,b,p,ef,el);
    uint i=((b*p.height+y)*p.width+x)*4u;
    debug[i]=q1.x; debug[i+1u]=q1.y; debug[i+2u]=q2.x; debug[i+3u]=dw.x+dw.y;
}

kernel void d2s_smaa_resolve(device const uchar *src_u8 [[buffer(0)]], device const float *src_f32 [[buffer(1)]],
                             device const uchar *weights [[buffer(2)]], device uchar *out_u8 [[buffer(3)]],
                             device float *out_f32 [[buffer(4)]], constant SmaaParams &p [[buffer(5)]],
                             uint3 gid [[thread_position_in_grid]]) {
    uint output_width=p.half_sbs!=0u?p.width/2u:p.width;
    uint x=gid.x,y=gid.y,b=gid.z; if(x>=output_width||y>=p.height||b>=p.batches)return;
    if((p.reserved&1u) != 0u) y=p.height-1u-y;
    uint eye=p.eye_width==0u?p.width:p.eye_width; uint ef=(x/eye)*eye, el=min(ef+eye,p.width)-1u;
    float4 c;
    if(p.half_sbs!=0u) {
        uint halfEye=eye/2u; uint halfTotal=halfEye*2u;
        uint he=min(x,halfTotal-1u); uint sourceX=(he<halfEye)?he*2u:eye+(he-halfEye)*2u;
        uint firstEye=(he<halfEye)?0u:eye; uint lastEye=firstEye+eye-1u;
        float4 a=smaa_resolve_at(src_u8,src_f32,weights,sourceX,y,b,p,firstEye,lastEye);
        float4 z=smaa_resolve_at(src_u8,src_f32,weights,min(sourceX+1u,lastEye),y,b,p,firstEye,lastEye);
        c=float4(smaa_encode(0.5f*(smaa_decode(a.rgb)+smaa_decode(z.rgb))),
                 0.5f*(a.a+z.a));
    } else c=smaa_resolve_at(src_u8,src_f32,weights,x,y,b,p,ef,el);
    for(uint ch=0;ch<p.channels;ch++) {
        uint si=smaa_dst_index(x,y,b,ch,p);
        float v=ch==0u?c.r:ch==1u?c.g:ch==2u?c.b:c.a;
        if(p.layout==0u||p.layout==2u) out_u8[si]=uchar(clamp(floor(v*255.0f+0.5f),0.0f,255.0f));
        else out_f32[si]=v;
    }
}
)MSL";
