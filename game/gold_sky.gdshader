shader_type sky;
uniform sampler2D foil : repeat_enable, filter_linear_mipmap;
uniform sampler2D cloud_mask : repeat_enable, filter_linear;
uniform float animation_time = -1.0;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float crease(float distance, float along, vec2 key) {
 key.x=mod(key.x,144.0);
 float strength=smoothstep(0.3,0.9,hash(key));
 strength*=mix(0.15+hash(key+8.0)*0.85,0.15+hash(key+19.0)*0.85,smoothstep(0.0,1.0,along));
 float direction=hash(key+41.0)>0.5 ? 1.0 : -1.0;
 return strength*(exp(-abs(distance-direction*0.012)*140.0)*0.07-exp(-abs(distance)*170.0)*0.05);
}
float brush(vec2 point, vec2 key, float roughness) {
 float result=0.0;
 for(int i=0;i<5;i++) {
  vec2 seed=key+float(i)*13.0;
  vec2 center=vec2(hash(seed+2.0),hash(seed+5.0))-0.5;
  float angle=(hash(seed+9.0)-0.5)*2.4;
  mat2 rotation=mat2(vec2(cos(angle),sin(angle)),vec2(-sin(angle),cos(angle)));
  vec2 q=rotation*(point-center);
  vec2 size=vec2(0.16+hash(seed+15.0)*0.22,0.055+hash(seed+21.0)*0.08);
  float edge=length(q/size)+(roughness-0.5)*0.55;
  float coverage=1.0-smoothstep(0.7,1.05,edge);
  float bristles=0.8+0.2*sin(q.y*310.0+roughness*4.0);
  result+=coverage*bristles*(hash(seed+31.0)-0.5);
 }
 return result;
}
void sky() {
 vec2 uv=SKY_COORDS;
 float seconds=animation_time<0.0 ? TIME : animation_time;
 vec2 tile=uv*vec2(144.0,72.0), cell=floor(tile), f=fract(tile);
 // Every foil face has its own rotated, translated brush patch. Short
 // irregular dabs replace long directional lines and global shared noise.
 float angle=hash(cell+3.0)*6.2831853;
 mat2 rotation=mat2(vec2(cos(angle),sin(angle)),vec2(-sin(angle),cos(angle)));
 vec2 local=rotation*(f-0.5);
 vec2 offset=vec2(hash(cell+7.0),hash(cell+23.0));
 vec2 patch_uv=local*vec2(0.12,0.08)+offset;
 float paint=texture(foil,patch_uv).r;
 float dabs=texture(foil,local*vec2(0.27,0.19)+offset+0.37).r;
 float pigment=0.5+(paint-0.5)*0.24+brush(local,cell,dabs)*0.65;
 float grain=texture(foil,local*1.7+offset).r-0.5;
 float dx=texture(foil,patch_uv+vec2(0.001,0.0)).r-paint;
 float dy=texture(foil,patch_uv+vec2(0.0,0.001)).r-paint;
 vec3 n=-normalize(EYEDIR);
 vec3 tangent=normalize(cross(vec3(0.0,1.0,0.0),n)+vec3(0.00001,0.0,0.0));
 vec3 bitangent=cross(n,tangent);
 vec2 fold=(vec2(hash(cell),hash(cell+17.0))-0.5)*0.035;
 vec3 normal=normalize(n+tangent*(dx*0.5+fold.x)+bitangent*(dy*0.5+fold.y));
 vec3 reflected=reflect(normalize(EYEDIR),normal);
 float broad=pow(max(0.0,dot(reflected,normalize(vec3(-0.35,0.45,0.82)))),3.0);
 float second=pow(max(0.0,dot(reflected,normalize(vec3(0.8,-0.1,-0.6)))),5.0);
 vec3 gold=mix(vec3(0.68,0.39,0.10),vec3(0.88,0.67,0.26),0.25+pigment*0.65);
 gold*=0.94+hash(cell+59.0)*0.10+grain*0.018;
 gold+=vec3(1.0,0.76,0.34)*(dx-dy)*0.5;
 gold+=vec3(0.13,0.10,0.04)*(broad+second*0.42);
 float edge_x=f.x<0.5 ? f.x : f.x-1.0;
 float edge_y=f.y<0.5 ? f.y : f.y-1.0;
 gold+=vec3(1.0,0.76,0.34)*(crease(edge_x,f.y,cell+vec2(f.x<0.5 ? 0.0 : 1.0,0.0))+crease(edge_y,f.x,cell.yx+vec2(f.y<0.5 ? 0.0 : 1.0,43.0)));
 float central=exp(-pow((uv.y-0.635)/0.13,2.0));
 gold+=vec3(0.15,0.12,0.045)*central;
 float cloud_u=fract(uv.x-seconds*0.0013);
 float cloud_y=(uv.y-0.42)/0.25;
 float cloud=texture(cloud_mask,vec2(cloud_u,clamp(cloud_y,0.0,1.0))).a;
 cloud*=step(0.0,cloud_y)*step(cloud_y,1.0);
 float tone=0.5+0.5*cos(cloud_u*6.0*6.2831853);
 gold=mix(gold,mix(vec3(0.16,0.14,0.10),vec3(0.94,0.87,0.68),tone),cloud*0.86);
 float upper=mix(0.48,1.0,smoothstep(0.46,0.66,uv.y));
 float lower=1.0-smoothstep(0.66,0.84,uv.y);
 float pole=smoothstep(0.025,0.14,uv.y);
 vec3 background=mix(vec3(0.025,0.017,0.009),gold,upper*lower*pole);
 vec2 sun=vec2(fract(uv.x-seconds*0.003-0.045+0.5)-0.5,uv.y-0.55);
 sun.x*=2.0;
 float disc=1.0-smoothstep(0.021,0.022,length(sun));
 // White sunlight is independent of the golden surface's shadow/tint.
 COLOR=mix(background,vec3(1.0),disc);
}
