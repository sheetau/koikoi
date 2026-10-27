extends RigidBody3D
## One physical card. Local +Z is its printed face, local +Y is its top edge.

const SIZE = Vector3(0.64, 1.04, 0.045)
var identity := -1
var slot := -1
var seat := -1
var surface: MeshInstance3D
var flight := false
var flight_from := Vector3.ZERO
var flight_to := Vector3.ZERO
var rotation_from := Quaternion.IDENTITY
var rotation_to := Quaternion.IDENTITY
var elapsed := 0.0
var duration := 0.6
var lift := 0.7
var orbit := false
var orbit_center := Vector3.ZERO
var orbit_angle := 0.0
var done := false

static func material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = color
	mat.roughness = 1.0
	return mat

func _ready() -> void:
	freeze = true
	mass = 0.014
	gravity_scale = 1.0
	linear_damp = 0.8
	angular_damp = 3.0
	collision_layer = 2
	collision_mask = 1
	var bounce := PhysicsMaterial.new()
	bounce.bounce = 0.55
	bounce.friction = 0.65
	physics_material_override = bounce
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = SIZE
	shape.shape = box
	add_child(shape)
	var edge := MeshInstance3D.new()
	var body := BoxMesh.new()
	body.size = SIZE
	edge.mesh = body
	edge.material_override = material(Color("b44519"))
	add_child(edge)
	# Fine dark outlines around each physical slab, including the stack sides.
	var outline := MeshInstance3D.new()
	var lines := ImmediateMesh.new()
	lines.surface_begin(Mesh.PRIMITIVE_LINES)
	var corners: Array[Vector3] = []
	for z in [-1, 1]:
		for y in [-1, 1]:
			for x in [-1, 1]:
				corners.append(Vector3(x, y, z) * SIZE * 0.501)
	for pair in [[0, 1], [0, 2], [1, 3], [2, 3], [4, 5], [4, 6], [5, 7], [6, 7], [0, 4], [1, 5], [2, 6], [3, 7]]:
		lines.surface_add_vertex(corners[pair[0]])
		lines.surface_add_vertex(corners[pair[1]])
	lines.surface_end()
	outline.mesh = lines
	outline.material_override = material(Color("21151b"))
	add_child(outline)
	var back := MeshInstance3D.new()
	var back_mesh := QuadMesh.new()
	back_mesh.size = Vector2(SIZE.x - 0.025, SIZE.y - 0.025)
	back.mesh = back_mesh
	back.position.z = -SIZE.z / 2.0 - 0.001
	back.rotation.y = PI
	back.material_override = material(Color("b44519"))
	add_child(back)
	surface = MeshInstance3D.new()
	var front := QuadMesh.new()
	front.size = Vector2(SIZE.x - 0.025, SIZE.y - 0.025)
	surface.mesh = front
	surface.position.z = SIZE.z / 2.0 + 0.001
	add_child(surface)
	set_face(null)

func set_face(texture: Texture2D) -> void:
	var mat := material(Color("b44519"))
	if texture != null:
		mat.albedo_color = Color.WHITE
		mat.albedo_texture = texture
		mat.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	surface.material_override = mat

func move_to(target: Vector3, angles: Vector3, seconds: float, arc: float = 0.6, circle: bool = false) -> void:
	freeze = true
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	flight_from = position
	flight_to = target
	rotation_from = quaternion
	rotation_to = Quaternion.from_euler(angles)
	duration = maxf(seconds, 0.015)
	elapsed = 0.0
	lift = arc
	orbit = circle
	orbit_center = (flight_from + flight_to) * 0.5
	orbit_angle = atan2(flight_from.z - orbit_center.z, flight_from.x - orbit_center.x)
	flight = true
	done = false

func _physics_process(delta: float) -> void:
	if not flight:
		return
	elapsed += delta
	var t := clampf(elapsed / duration, 0.0, 1.0)
	# Minimum-jerk trajectory has zero velocity and acceleration at both ends.
	var u := t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
	position = flight_from.lerp(flight_to, u)
	if orbit:
		var radius := Vector2(flight_from.x - orbit_center.x, flight_from.z - orbit_center.z).length()
		var angle := orbit_angle + PI * u
		position.x = orbit_center.x + cos(angle) * radius
		position.z = orbit_center.z + sin(angle) * radius
	position.y += sin(PI * u) * lift
	quaternion = rotation_from.slerp(rotation_to, u)
	if t >= 1.0:
		position = flight_to
		quaternion = rotation_to
		flight = false
		done = true

func strike() -> void:
	# Actual rigid-body collision against the tabletop, followed by guided pickup.
	flight = false
	freeze = false
	sleeping = false
	linear_velocity = Vector3(0.12, -4.8, 0.08)
	angular_velocity = Vector3(0.45, 0.0, 0.25)
