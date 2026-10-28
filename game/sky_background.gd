extends Node
var clouds: Node2D
var cloud_viewport: SubViewport
var foil: NoiseTexture2D
func configure(material: ShaderMaterial) -> void:
	var noise := FastNoiseLite.new()
	noise.seed = 37851
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	noise.frequency = 0.035
	noise.fractal_octaves = 4
	noise.fractal_gain = 0.45
	foil = NoiseTexture2D.new()
	foil.width = 1024
	foil.height = 1024
	foil.seamless = true
	foil.noise = noise
	material.set_shader_parameter("foil",foil)
	cloud_viewport = SubViewport.new()
	cloud_viewport.size = Vector2i(4096,512)
	cloud_viewport.transparent_bg = true
	cloud_viewport.disable_3d = true
	cloud_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(cloud_viewport)
	clouds = preload("res://sky_clouds.gd").new()
	cloud_viewport.add_child(clouds)
	material.set_shader_parameter("cloud_mask",cloud_viewport.get_texture())
