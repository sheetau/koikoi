shader_type sky;
uniform sampler2D foil_0 : repeat_disable, filter_linear_mipmap;
uniform sampler2D foil_1 : repeat_disable, filter_linear_mipmap;
uniform sampler2D foil_2 : repeat_disable, filter_linear_mipmap;
uniform sampler2D foil_3 : repeat_disable, filter_linear_mipmap;
uniform sampler2D foil_4 : repeat_disable, filter_linear_mipmap;
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
// Read the supplied images as texture data. Their original olive/brown tint
// does not replace the existing gold palette. Center each image independently.
float foil_height(vec2 uv, float variant) {
 vec3 sample_color;
 float mean;
 if(variant<1.0) { sample_color=texture(foil_0,uv).rgb; mean=0.4390; }
 else if(variant<2.0) { sample_color=texture(foil_1,uv).rgb; mean=0.5082; }
 else if(variant<3.0) { sample_color=texture(foil_2,uv).rgb; mean=0.4139; }
 else if(variant<4.0) { sample_color=texture(foil_3,uv).rgb; mean=0.5300; }
 else { sample_color=texture(foil_4,uv).rgb; mean=0.4417; }
 return 0.5+dot(sample_color,vec3(0.299,0.587,0.114))-mean;
}
void sky() {
 vec2 uv=SKY_COORDS;
 float seconds=animation_time<0.0 ? TIME : animation_time;
 vec2 tile=uv*vec2(144.0,72.0), cell=floor(tile), f=fract(tile);
 // Normalized square UVs stretch each rectangular source to a square.
 // A seeded image choice, rotation and crop offset vary from face to face.
 // Keep the rotated crop inside the source to avoid wrapping visible seams.
 float variant=floor(hash(cell+83.0)*5.0);
 float angle=hash(cell+3.0)*6.2831853;
 mat2 rotation=mat2(vec2(cos(angle),sin(angle)),vec2(-sin(angle),cos(angle)));
 float crop=0.78/(abs(cos(angle))+abs(sin(angle)));
 vec2 offset=(vec2(hash(cell+7.0),hash(cell+23.0))-0.5)*0.18;
 vec2 patch_uv=rotation*(f-0.5)*crop+0.5+offset;
 float paint=foil_height(patch_uv,variant);
 float pigment=0.5+(paint-0.5)*0.24;
 float dx=foil_height(patch_uv+vec2(0.004,0.0),variant)-paint;
 float dy=foil_height(patch_uv+vec2(0.0,0.004),variant)-paint;
 vec3 n=-normalize(EYEDIR);
 vec3 tangent=normalize(cross(vec3(0.0,1.0,0.0),n)+vec3(0.00001,0.0,0.0));
 vec3 bitangent=cross(n,tangent);
 vec2 fold=(vec2(hash(cell),hash(cell+17.0))-0.5)*0.035;
 vec3 normal=normalize(n+tangent*(dx*0.5+fold.x)+bitangent*(dy*0.5+fold.y));
 vec3 reflected=reflect(normalize(EYEDIR),normal);
 float broad=pow(max(0.0,dot(reflected,normalize(vec3(-0.35,0.45,0.82)))),3.0);
 float second=pow(max(0.0,dot(reflected,normalize(vec3(0.8,-0.1,-0.6)))),5.0);
 vec3 gold=mix(vec3(0.68,0.39,0.10),vec3(0.88,0.67,0.26),0.25+pigment*0.65);
 gold*=(0.94+hash(cell+59.0)*0.10)*(1.0+(paint-0.5)*0.8);
 gold+=vec3(1.0,0.76,0.34)*(dx-dy)*0.10;
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
