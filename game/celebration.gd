extends Node3D
## Presentation-only sparks and diffuse pressure waves; no physics bodies.
var sparks: Array = []
var mesh := ImmediateMesh.new()
var rng := RandomNumberGenerator.new()
var emitted := 0
var impacts := 0
func _ready() -> void:
	rng.seed = 47113
	var node := MeshInstance3D.new()
	node.mesh = mesh
	var material := StandardMaterial3D.new()
	material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	material.vertex_color_use_as_albedo = true
	material.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
	material.blend_mode = BaseMaterial3D.BLEND_MODE_ADD
	node.material_override = material
	add_child(node)
func emit_at(pose: Transform3D, amount: int, burst: bool = false) -> void:
	emitted += amount
	for i in range(amount):
		var edge := Vector3(rng.randf_range(-0.42,0.42),rng.randf_range(-0.67,0.67),0)
		var direction := Vector3(rng.randf_range(-1,1),rng.randf_range(-1,1),rng.randf_range(-0.3,0.3)).normalized()
		sparks.append({"p":pose*edge,"v":pose.basis*direction*rng.randf_range(1.0,3.5 if burst else 1.4),"age":0.0,"life":rng.randf_range(0.25,0.75)})
	while sparks.size()>2400:
		sparks.pop_front()
func impact(pose: Transform3D) -> void:
	impacts += 1
	emit_at(pose,160,true)
	var flash := MeshInstance3D.new()
	var geometry := QuadMesh.new()
	geometry.size = Vector2(2.1,2.1)
	flash.mesh = geometry
	var material := ShaderMaterial.new()
	material.shader = preload("res://pressure_wave.gdshader")
	material.set_shader_parameter("seed",rng.randf()*100.0)
	flash.material_override = material
	add_child(flash)
	flash.transform = pose
	var tween := create_tween().set_parallel()
	tween.tween_property(flash,"scale",Vector3.ONE*2.2,0.42)
	tween.tween_method(func(value):material.set_shader_parameter("opacity",value),0.55,0.0,0.42)
	tween.chain().tween_callback(flash.queue_free)
func _process(delta: float) -> void:
	mesh.clear_surfaces()
	if sparks.is_empty(): return
	for i in range(sparks.size()-1,-1,-1):
		var s: Dictionary = sparks[i]
		s.age += delta
		if s.age>=s.life:
			sparks.remove_at(i)
			continue
		s.p += s.v*delta
		s.v += Vector3(0,-1.1,0)*delta
	if sparks.is_empty(): return
	mesh.surface_begin(Mesh.PRIMITIVE_LINES)
	for s in sparks:
		var alpha: float = 1.0-s.age/s.life
		mesh.surface_set_color(Color(1,0.68+alpha*0.25,0.27+alpha*0.5,alpha))
		mesh.surface_add_vertex(s.p)
		mesh.surface_add_vertex(s.p-s.v*0.035)
	mesh.surface_end()
