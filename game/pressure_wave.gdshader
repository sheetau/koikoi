shader_type spatial;
render_mode unshaded, blend_mix, depth_draw_never, cull_disabled;
uniform float opacity = 0.55;
uniform float seed = 0.0;
float hash(vec2 p) { return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p) {
 vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
 return mix(mix(hash(i),hash(i+vec2(1.0,0.0)),f.x),mix(hash(i+vec2(0.0,1.0)),hash(i+vec2(1.0)),f.x),f.y);
}
void fragment() {
 vec2 p=(UV-0.5)*2.0;
 float cloud=noise(p*5.0+seed)*0.65+noise(p*13.0+seed)*0.35;
 float radius=length(p);
 float wave=exp(-pow((radius-0.53-(cloud-0.5)*0.16)/0.15,2.0));
 float pressure=exp(-pow((radius-0.46)/0.035,2.0))*0.15;
 ALBEDO=mix(vec3(0.76,0.74,0.68),vec3(1.0,0.99,0.94),cloud);
 ALPHA=(wave*(0.2+cloud*0.8)+pressure)*opacity*smoothstep(1.0,0.78,radius);
}
