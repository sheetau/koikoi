shader_type sky;
uniform float view_center = 0.68;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float stroke(vec2 p, vec2 a, vec2 b) {
 vec2 axis=b-a;
 return length(p-a-axis*clamp(dot(p-a,axis)/dot(axis,axis),0.0,1.0))-0.13;
}
float cloud_shape(vec2 p, int kind) {
 float d=10.0;
 if(kind==0) {
  // Rounded U, with unequal arms like a traditional cloud scroll.
  d=min(stroke(p,vec2(-1.0,-0.65),vec2(-1.0,0.7)),stroke(p,vec2(-1.0,0.7),vec2(0.9,0.7)));
  d=min(d,stroke(p,vec2(0.9,0.7),vec2(0.9,-0.25)));
 } else if(kind==1) {
  // S: opposing half circles linked by long horizontal strokes.
  d=min(stroke(p,vec2(-0.6,-0.8),vec2(1.1,-0.8)),stroke(p,vec2(-0.6,0.0),vec2(0.6,0.0)));
  d=min(d,stroke(p,vec2(-1.1,0.8),vec2(0.6,0.8)));
  if(p.x< -0.6) d=min(d,abs(length(p-vec2(-0.6,-0.4))-0.4)-0.13);
  if(p.x> 0.6) d=min(d,abs(length(p-vec2(0.6,0.4))-0.4)-0.13);
 } else {
  // The broad upper/lower bars and narrow bridge of an e-gumo.
  d=min(stroke(p,vec2(-1.15,-0.7),vec2(0.9,-0.7)),stroke(p,vec2(-0.9,0.7),vec2(1.15,0.7)));
  d=min(d,stroke(p,vec2(0.0,-0.7),vec2(0.0,0.7)));
 }
 return d;
}
void sky() {
	vec2 uv=SKY_COORDS;
	vec2 tile=uv*vec2(144.0,72.0);
	vec2 cell=floor(tile);
	vec2 f=fract(tile);
	float seam=smoothstep(0.0,0.016,min(min(f.x,1.0-f.x),min(f.y,1.0-f.y)));
	float fold=0.035*cos(f.x*3.14159265);
	float grain=hash(floor(uv*vec2(4000.0,2000.0)))*0.025;
	float distance_from_center=clamp(abs(uv.y-view_center)/0.14,0.0,1.0);
	float light=0.94-0.88*pow(distance_from_center,1.4);
	vec3 gold=mix(vec3(0.10,0.06,0.025),vec3(0.83,0.61,0.19),light);
	gold*=0.94+hash(cell)*0.10+fold+grain;
	gold*=mix(0.94,1.0,seam);
	// Broad reflected light, offset by the shallow normal and roughness of
	// each foil facet. Smooth lobes avoid a plastic, uniformly shiny surface.
	vec2 normal_offset=(vec2(hash(cell),hash(cell+19.0))-0.5)*0.025;
	float reflected_x=fract(uv.x+normal_offset.x+0.5)-0.5;
	float roughness=0.12+hash(cell+37.0)*0.06;
	float reflected=exp(-pow((reflected_x-0.045)/roughness,2.0)-pow((uv.y+normal_offset.y-0.64)/0.09,2.0));
	gold+=vec3(0.16,0.12,0.055)*reflected*(0.55+0.45*cos(f.x*3.14159265));
	vec2 sun=vec2(fract(uv.x-TIME*0.003-0.045+0.5)-0.5,uv.y-0.59);
	sun.x*=2.0;
	float disc=1.0-smoothstep(0.021,0.022,length(sun));
	gold=mix(gold,vec3(0.98,0.92,0.73),disc*0.94);
	for(int i=0;i<8;i++) {
		float n=float(i);
		vec2 p=vec2(fract(uv.x-TIME*0.0013-n*0.137+0.5)-0.5,uv.y-(0.60+0.022*sin(n*2.8)));
		p/=vec2(0.035+0.009*hash(vec2(n,2.0)),0.010);
		float cloud=1.0-smoothstep(-0.025,0.025,cloud_shape(p,i%3));
		float tone=0.5+0.5*cos(p.x*7.5+n);
		vec3 cloud_color=mix(vec3(0.16,0.14,0.10),vec3(0.94,0.87,0.68),tone);
		gold=mix(gold,cloud_color,cloud*0.82);
	}
	float edge_fade=1.0-smoothstep(0.065,0.135,abs(uv.y-view_center));
	float pole_fade=smoothstep(0.04,0.14,uv.y)*smoothstep(0.04,0.14,1.0-uv.y);
	COLOR=mix(vec3(0.025,0.017,0.009),gold,edge_fade*pole_fade);
}
