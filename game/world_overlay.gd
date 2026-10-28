extends Node3D
## Ordinary Godot controls rendered and hit-tested on a plane in the world.
var SIZE = Vector2(6.15, 5.85)
var PIXELS = Vector2i(820, 780)
var viewport: SubViewport
var surface: MeshInstance3D

func _ready() -> void:
	viewport = SubViewport.new()
	viewport.size = PIXELS
	viewport.transparent_bg = true
	viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	viewport.gui_disable_input = false
	viewport.gui_embed_subwindows = true
	add_child(viewport)
	surface = MeshInstance3D.new()
	var mesh := QuadMesh.new()
	mesh.size = SIZE
	surface.mesh = mesh
	surface.layers = 4
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	mat.depth_draw_mode = BaseMaterial3D.DEPTH_DRAW_DISABLED
	mat.cull_mode = BaseMaterial3D.CULL_DISABLED
	mat.albedo_texture = viewport.get_texture()
	mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR
	surface.material_override = mat
	var fringe := ShaderMaterial.new()
	fringe.shader = preload("res://chromatic_surface.gdshader")
	fringe.set_shader_parameter("source_tex",viewport.get_texture())
	fringe.set_shader_parameter("texel",Vector2.ONE/Vector2(PIXELS))
	mat.next_pass = fringe
	add_child(surface)

func fit_content(pixels: Vector2i, _camera: Camera3D) -> void:
	PIXELS = pixels
	SIZE = Vector2(pixels) * 0.0075
	viewport.size = pixels
	(surface.mesh as QuadMesh).size = SIZE
	surface.material_override.next_pass.set_shader_parameter("texel",Vector2.ONE/Vector2(pixels))
	scale = Vector3.ONE # World UI keeps its size and anchor when the camera zooms.

func screen_point(local_pixel: Vector2, camera: Camera3D) -> Vector2:
	var pos := Vector3((local_pixel.x / PIXELS.x - 0.5) * SIZE.x, (0.5 - local_pixel.y / PIXELS.y) * SIZE.y, 0.0)
	return camera.unproject_position(to_global(pos))

func route(event: InputEvent, camera: Camera3D) -> void:
	if event is InputEventMouse:
		var plane := Plane(global_basis.z, global_position)
		var point = plane.intersects_ray(camera.project_ray_origin(event.position), camera.project_ray_normal(event.position))
		if point == null:
			return
		var p := to_local(point)
		var copy = event.duplicate()
		copy.position = Vector2((p.x / SIZE.x + 0.5) * PIXELS.x, (0.5 - p.y / SIZE.y) * PIXELS.y)
		copy.global_position = copy.position
		viewport.push_input(copy, true)
	else:
		viewport.push_input(event, true)

func contains_screen(screen: Vector2, camera: Camera3D) -> bool:
	var point = Plane(global_basis.z,global_position).intersects_ray(camera.project_ray_origin(screen),camera.project_ray_normal(screen))
	if point == null:
		return false
	var local := to_local(point)
	return absf(local.x) <= SIZE.x/2 and absf(local.y) <= SIZE.y/2
