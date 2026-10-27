extends RigidBody3D
## Zero-thickness, two-sided artwork. The tiny collision proxy is not rendered.

const SIZE = Vector3(0.78, 1.28, 0.006)
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
var lift := 0.0
var clearance := PackedFloat32Array()
var orbit := false
var orbit_center := Vector3.ZERO
var orbit_angle := 0.0
var done := false
var collision_proxy: CollisionShape3D
var pair_peer = null
var pair_sign := 1.0
var reveal_turn := false

static func material(color: Color) -> StandardMaterial3D:
	var mat := StandardMaterial3D.new()
	mat.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
	mat.albedo_color = color
	mat.roughness = 1.0
	return mat

func _ready() -> void:
	freeze = true
	mass = 0.014
	continuous_cd = true
	axis_lock_angular_x = true
	axis_lock_angular_y = true
	axis_lock_angular_z = true
	linear_damp = 1.2
	angular_damp = 5.0
	collision_layer = 2
	collision_mask = 3
	var bounce := PhysicsMaterial.new()
	bounce.bounce = 0.38
	bounce.friction = 0.8
	physics_material_override = bounce
	collision_proxy = CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = SIZE
	box.margin = 0.001
	collision_proxy.shape = box
	add_child(collision_proxy)
	var back := MeshInstance3D.new()
	var quad := QuadMesh.new()
	quad.size = Vector2(SIZE.x, SIZE.y)
	back.mesh = quad
	back.rotation.y = PI
	back.material_override = material(Color("b44519"))
	add_child(back)
	surface = MeshInstance3D.new()
	surface.mesh = quad
	add_child(surface)
	set_face(null)

func set_face(texture: Texture2D) -> void:
	if texture == null:
		surface.material_override = material(Color("b44519"))
		return
	var mat := ShaderMaterial.new()
	mat.shader = preload("res://card_face.gdshader")
	mat.set_shader_parameter("artwork", texture)
	surface.material_override = mat

func stow() -> void:
	hide()
	collision_layer = 0
	collision_proxy.disabled = true
	flight = false
	freeze = true

func unstow() -> void:
	show()
	collision_layer = 2
	collision_proxy.disabled = false

func move_to(target: Vector3, angles: Vector3, seconds: float, arc: float = 0.0, circle: bool = false) -> void:
	freeze = true
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	flight_from = position
	flight_to = target
	rotation_from = quaternion
	rotation_to = Quaternion.from_euler(angles)
	duration = maxf(seconds, 0.025)
	elapsed = 0.0
	lift = arc
	clearance.resize(97)
	clearance.fill(0.0)
	orbit = circle
	orbit_center = (flight_from + flight_to) * 0.5
	orbit_angle = atan2(flight_from.z - orbit_center.z, flight_from.x - orbit_center.x)
	flight = true
	done = false

static func turn_between(a: Quaternion, b: Quaternion, weight: float) -> Quaternion:
	# A half turn has two equally short paths. Tiny physics roundoff must not
	# make two otherwise parallel cards select opposite rotational directions.
	if absf(a.dot(b)) < 0.001:
		var delta := a.inverse() * b
		var axis := Vector3(delta.x, delta.y, delta.z).normalized()
		if axis.x < -0.0001 or (absf(axis.x) <= 0.0001 and axis.y < -0.0001) or (absf(axis.x) <= 0.0001 and absf(axis.y) <= 0.0001 and axis.z < 0):
			axis = -axis
		return a * Quaternion(axis, PI * weight)
	return a.slerp(b, weight)

func pose_at(time: float) -> Transform3D:
	var t := clampf(time / duration, 0.0, 1.0)
	var u := t * t * (3.0 - 2.0 * t)
	var pos := flight_from.lerp(flight_to, u)
	if orbit:
		var radius := Vector2(flight_from.x - orbit_center.x, flight_from.z - orbit_center.z).length()
		var angle := orbit_angle + PI * u
		pos.x = orbit_center.x + cos(angle) * radius
		pos.z = orbit_center.z + sin(angle) * radius
	pos.y += 4.0 * u * (1.0 - u) * lift
	if clearance.size() == 97:
		var index := minf(t * 96.0, 95.99999)
		var cell := int(index)
		pos.y += lerpf(clearance[cell], clearance[cell + 1], index - cell)
	var q := turn_between(rotation_from, rotation_to, u)
	if reveal_turn:
		var facing := rotation_from * Quaternion(Vector3.UP, PI)
		q = turn_between(rotation_from, facing, u / 0.48) if u < 0.48 else turn_between(facing, rotation_to, (u - 0.48) / 0.52)
	var rotation_basis := Basis(q)
	if pair_peer != null and is_instance_valid(pair_peer):
		var peer_pose: Transform3D = pair_peer.pose_at(time)
		var relative: Vector3 = rotation_basis.inverse() * (pos - peer_pose.origin)
		if absf(relative.x) < SIZE.x + 0.01 and absf(relative.y) < SIZE.y + 0.01 and relative.z * pair_sign < 0.012:
			pos += rotation_basis.z * (0.012 * pair_sign - relative.z)
	return Transform3D(rotation_basis, pos)

func _physics_process(delta: float) -> void:
	if not flight:
		return
	elapsed += delta
	transform = pose_at(elapsed)
	if elapsed >= duration:
		position = flight_to
		quaternion = rotation_to
		flight = false
		done = true
		reveal_turn = false
		pair_peer = null
		pair_sign = 1.0

func strike() -> void:
	flight = false
	freeze = false
	sleeping = false
	linear_velocity = Vector3(0, -2.6, 0)
	angular_velocity = Vector3.ZERO
