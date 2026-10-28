shader_type sky;
uniform sampler2D foil : repeat_enable, filter_linear_mipmap;
uniform sampler2D cloud_mask : repeat_enable, filter_linear;
uniform float animation_time = -1.0;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
void sky() {
	vec2 uv=SKY_COORDS;
	float seconds=animation_time<0.0 ? TIME : animation_time;
	vec2 tile=uv*vec2(144.0,72.0);
	vec2 cell=floor(tile), f=fract(tile);
	vec2 grain_uv=uv*vec2(4.0,2.0);
	float height=texture(foil,grain_uv).r;
	float dx=texture(foil,grain_uv+vec2(0.001,0.0)).r-height;
	float dy=texture(foil,grain_uv+vec2(0.0,0.001)).r-height;
	float fine=texture(foil,grain_uv*3.0).r;
	float brush_a=texture(foil,uv*vec2(2.0,18.0)+vec2(0.13,0.31)).r;
	float brush_b=texture(foil,uv*vec2(20.0,1.0)+vec2(0.43,0.17)).r;
	float brush=mix(brush_a,brush_b,hash(cell+31.0));
	vec3 n=-normalize(EYEDIR);
	vec3 tangent=normalize(cross(vec3(0.0,1.0,0.0),n)+vec3(0.00001,0.0,0.0));
	vec3 bitangent=cross(n,tangent);
	vec2 fold=(vec2(hash(cell),hash(cell+17.0))-0.5)*0.045;
	vec3 normal=normalize(n+tangent*(dx*0.35+fold.x)+bitangent*(dy*0.35+fold.y));
	vec3 reflected=reflect(normalize(EYEDIR),normal);
	// Broad soft-box reflections in wrinkled metal; normals, seams and
	// roughness stay fixed to the sphere when the camera is moved.
	float broad=pow(max(0.0,dot(reflected,normalize(vec3(-0.35,0.45,0.82)))),3.0);
	float second=pow(max(0.0,dot(reflected,normalize(vec3(0.8,-0.1,-0.6)))),5.0);
	float rough=0.55+0.45*fine;
	// Warm broad pigment variation and directional brush grain, without
	// the dark high-frequency mottling of the earlier foil.
	float pigment=clamp(0.25+height*0.60+brush*0.15,0.0,1.0);
	vec3 gold=mix(vec3(0.66,0.36,0.075),vec3(0.86,0.65,0.24),pigment);
	gold*=0.91+hash(cell)*0.13+(brush-0.5)*0.10;
	gold+=vec3(0.18,0.14,0.065)*(broad+second*0.42)*rough;
	vec2 edge=min(f,1.0-f);
	float seam=exp(-min(edge.x,edge.y)*220.0);
	float rim=exp(-min(abs(f.x-0.018),abs(f.y-0.018))*180.0);
	gold=gold*(1.0-seam*0.17)+vec3(0.15,0.12,0.055)*rim*(0.5+broad);
	vec2 sun=vec2(fract(uv.x-seconds*0.003-0.045+0.5)-0.5,uv.y-0.57);
	sun.x*=2.0;
	float disc=1.0-smoothstep(0.021,0.022,length(sun));
	gold=mix(gold,vec3(0.98,0.92,0.73),disc*0.94);
	float cloud_u=fract(uv.x-seconds*0.0013);
	float cloud_y=(uv.y-0.42)/0.25;
	float cloud=texture(cloud_mask,vec2(cloud_u,clamp(cloud_y,0.0,1.0))).a;
	cloud*=step(0.0,cloud_y)*step(cloud_y,1.0);
	// One broad light/dark cycle per sixth of the panorama: five times
	// the previous gradient wavelength, independent of each U's width.
	float tone=0.5+0.5*cos(cloud_u*6.0*6.2831853);
	gold=mix(gold,mix(vec3(0.16,0.14,0.10),vec3(0.94,0.87,0.68),tone),cloud*0.86);
	// Fixed world latitudes, with a restrained upper shade and deep lower
	// falloff. Pole patterns disappear without a camera-following vignette.
	float upper=mix(0.48,1.0,smoothstep(0.46,0.66,uv.y));
	float lower=1.0-smoothstep(0.66,0.84,uv.y);
	float pole=smoothstep(0.025,0.14,uv.y);
	COLOR=mix(vec3(0.025,0.017,0.009),gold,upper*lower*pole);
}
