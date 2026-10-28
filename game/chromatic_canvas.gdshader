shader_type canvas_item;
varying vec4 tint;
void vertex() { tint = COLOR; }
void fragment() {
	vec2 shift = vec2(TEXTURE_PIXEL_SIZE.x * 0.65, 0.0);
	vec4 c = texture(TEXTURE, UV);
	vec4 r = texture(TEXTURE, UV + shift);
	vec4 b = texture(TEXTURE, UV - shift);
	float a = max(c.a, max(r.a, b.a));
	COLOR = vec4(vec3(r.r*r.a, c.g*c.a, b.b*b.a)/max(a,0.001),a)*tint;
}
