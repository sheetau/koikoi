shader_type spatial;
render_mode unshaded;
uniform sampler2D artwork : source_color, filter_linear_mipmap_anisotropic;
uniform vec4 frame_color : source_color = vec4(0.705882, 0.270588, 0.098039, 1.0);
void fragment() {
    // Commons artwork uses a 976 x 1600 viewbox with a 52-unit outer frame.
    // Mask this border geometrically; preserve all black illustration ink.
    bool border = UV.x < 53.0 / 976.0 || UV.x > 923.0 / 976.0
        || UV.y < 53.0 / 1600.0 || UV.y > 1547.0 / 1600.0;
    ALBEDO = border ? frame_color.rgb : texture(artwork, UV).rgb;
}
