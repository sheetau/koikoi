shader_type spatial;
render_mode unshaded, blend_mix, depth_draw_never, cull_disabled;
uniform sampler2D source_tex : filter_linear;
uniform vec2 texel = vec2(0.001);
void fragment() {
	vec4 c=texture(source_tex,UV);
	vec4 r=texture(source_tex,UV+vec2(texel.x*0.7,0.0));
	vec4 b=texture(source_tex,UV-vec2(texel.x*0.7,0.0));
	float red=max(0.0,r.a-c.a);
	float blue=max(0.0,b.a-c.a);
	ALBEDO=vec3(red,blue*0.55,blue)/max(0.001,max(red,blue));
	ALPHA=max(red,blue)*0.55;
}
