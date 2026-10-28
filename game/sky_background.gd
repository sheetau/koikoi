extends Node
var clouds: Node2D
var cloud_viewport: SubViewport
const FOIL_TILES: Array[Texture2D] = [
	preload("res://assets/sky/kinbyobu-tile-1.png"),
	preload("res://assets/sky/kinbyobu-tile-2.png"),
	preload("res://assets/sky/kinbyobu-tile-3.png"),
	preload("res://assets/sky/kinbyobu-tile-4.png"),
	preload("res://assets/sky/kinbyobu-tile-5.png"),
]
func configure(material: ShaderMaterial) -> void:
	for i in range(FOIL_TILES.size()):
		material.set_shader_parameter("foil_%d"%i,FOIL_TILES[i])
	cloud_viewport = SubViewport.new()
	cloud_viewport.size = Vector2i(4096,512)
	cloud_viewport.transparent_bg = true
	cloud_viewport.disable_3d = true
	cloud_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	add_child(cloud_viewport)
	clouds = preload("res://sky_clouds.gd").new()
	cloud_viewport.add_child(clouds)
	material.set_shader_parameter("cloud_mask",cloud_viewport.get_texture())
