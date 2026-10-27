extends RigidBody3D
## Zero-thickness, two-sided artwork. The tiny collision proxy is not rendered.

signal arrived(card)

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
var start_delay := 0.0
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
var kick_velocity := NAN
var returning := false
var bottom_return := false
var capture_slide := false
var gather_group := -1
var arrival_sound := ""
var return_curve := false
var return_c1 := Vector3.ZERO
var return_c2 := Vector3.ZERO
var return_lengths := PackedFloat32Array()
var return_length := 0.0
var role_route := false
var role_side := 1.0
var role_edge := 6.2
var role_lengths := Vector3.ONE

static func overlaps(a: Transform3D, b: Transform3D, margin: float = 0.015) -> bool:
	var axes: Array[Vector3] = [a.basis.x, a.basis.y, a.basis.z, b.basis.x, b.basis.y, b.basis.z]
	for x in [a.basis.x, a.basis.y, a.basis.z]:
		for y in [b.basis.x, b.basis.y, b.basis.z]:
			if x.cross(y).length_squared() > 0.00001:
				axes.append(x.cross(y).normalized())
	var half := SIZE / 2 + Vector3.ONE * margin
	for axis in axes:
		var ar: float = absf(axis.dot(a.basis.x)) * half.x + absf(axis.dot(a.basis.y)) * half.y + absf(axis.dot(a.basis.z)) * half.z
		var br: float = absf(axis.dot(b.basis.x)) * half.x + absf(axis.dot(b.basis.y)) * half.y + absf(axis.dot(b.basis.z)) * half.z
		if absf(axis.dot(a.origin - b.origin)) >= ar + br:
			return false
	return true

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
	gather_group = -1
	arrival_sound = ""

func unstow() -> void:
	show()
	collision_layer = 2
	collision_proxy.disabled = false

func move_to(target: Vector3, angles: Vector3, seconds: float, arc: float = 0.0, circle: bool = false) -> void:
	returning = false
	return_curve = false
	bottom_return = false
	gather_group = -1
	arrival_sound = ""
	capture_slide = false
	role_route = false
	freeze = true
	linear_velocity = Vector3.ZERO
	angular_velocity = Vector3.ZERO
	flight_from = position
	flight_to = target
	rotation_from = quaternion
	rotation_to = Quaternion.from_euler(angles)
	start_delay = 0.0
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

func route_to_role(player: int, board_edge: float = 6.5) -> void:
	role_route = true
	role_side = -1.0 if player == 0 else 1.0
	role_edge = role_side * maxf(board_edge, maxf(absf(flight_from.x) + 0.8, flight_to.x * role_side + 1.0))
	var radius := absf(flight_to.z - flight_from.z) / 2
	role_lengths = Vector3(absf(role_edge - flight_from.x), PI * sqrt((1.0 + radius * radius) / 2), absf(flight_to.x - role_edge))

func stock_point(u: float) -> Vector3:
	return flight_from.bezier_interpolate(return_c1,return_c2,flight_to,u)

func route_to_stock(straight: bool = false) -> void:
	return_curve = true
	return_c1 = flight_from + Vector3(2.0 if flight_from.x <= flight_to.x else -2.0,0,0)
	return_c2 = flight_to + Vector3(2.2,0,signf(flight_from.z- flight_to.z)*1.5)
	if straight:
		return_c1 = flight_from.lerp(flight_to,1.0/3)
		return_c2 = flight_from.lerp(flight_to,2.0/3)
	return_lengths = PackedFloat32Array([0.0])
	return_length = 0
	var previous := flight_from
	for i in range(1,65):
		var point := stock_point(i/64.0)
		return_length += point.distance_to(previous)
		return_lengths.append(return_length)
		previous = point

func stock_parameter(distance: float) -> float:
	for i in range(1,return_lengths.size()):
		if return_lengths[i] >= distance:
			return (i-1+inverse_lerp(return_lengths[i-1],return_lengths[i],distance))/64.0
	return 1.0

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

func role_clearance_weight(t: float) -> float:
	var u := clampf(t, 0.0, 1.0)
	var distance := u * (role_lengths.x + role_lengths.y + role_lengths.z)
	return 1.0 - smoothstep(role_lengths.x, role_lengths.x + role_lengths.y / 2, distance)

func pose_at(time: float) -> Transform3D:
	if time < start_delay:
		return Transform3D(Basis(rotation_from), flight_from)
	var clock_t := clampf(time / duration, 0.0, 1.0)
	var t := clampf((time - start_delay) / (duration - start_delay), 0.0, 1.0)
	var u := t if role_route or return_curve else t * t * (3.0 - 2.0 * t)
	var pos := flight_from.lerp(flight_to, u)
	if return_curve:
		var curve_t := stock_parameter(t*return_length)
		pos = stock_point(curve_t)
		pos.y = lerpf(flight_from.y,flight_to.y,smoothstep(0.0,0.65,curve_t))
	if orbit:
		var radius := Vector2(flight_from.x - orbit_center.x, flight_from.z - orbit_center.z).length()
		var angle := orbit_angle - PI * u
		pos.x = orbit_center.x + cos(angle) * radius
		pos.z = orbit_center.z + sin(angle) * radius
	var role_turn := u
	if role_route:
		var distance := u * (role_lengths.x + role_lengths.y + role_lengths.z)
		if distance < role_lengths.x:
			pos.x = lerpf(flight_from.x, role_edge, distance / role_lengths.x)
			pos.z = flight_from.z
		elif distance < role_lengths.x + role_lengths.y:
			var angle := (distance - role_lengths.x) / role_lengths.y * PI
			pos.x = role_edge + role_side * sin(angle)
			pos.z = lerpf(flight_from.z, flight_to.z, (1.0 - cos(angle)) / 2)
		else:
			pos.x = lerpf(role_edge, flight_to.x, (distance - role_lengths.x - role_lengths.y) / role_lengths.z)
			pos.z = flight_to.z
		role_turn = smoothstep(role_lengths.x, role_lengths.x + role_lengths.y / 2, distance)
		pos.y = lerpf(flight_from.y, flight_to.y, role_turn)
	var extra_y := 4.0 * u * (1.0 - u) * lift
	if clearance.size() == 97:
		var index := minf(clock_t * 96.0, 95.99999)
		var cell := int(index)
		extra_y += lerpf(clearance[cell], clearance[cell + 1], index - cell) * (role_clearance_weight(t) if role_route else 1.0)
	var q := turn_between(rotation_from, rotation_to, role_turn)
	if reveal_turn:
		var facing := rotation_from * Quaternion(Vector3.UP, PI)
		q = turn_between(rotation_from, facing, u / 0.48) if u < 0.48 else turn_between(facing, rotation_to, (u - 0.48) / 0.52)
	if returning:
		# Direct translation starts with the rotation; finish folding before
		# parallel cards converge into adjacent stock layers.
		q = turn_between(rotation_from, rotation_to, smoothstep(0.0, 0.25, t))
	var rotation_basis := Basis(q)
	pos.y += extra_y
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
		arrived.emit(self)

func _integrate_forces(state: PhysicsDirectBodyState3D) -> void:
	if not is_nan(kick_velocity):
		state.linear_velocity = Vector3(0, kick_velocity, 0)
		kick_velocity = NAN

func kick(velocity: float) -> void:
	flight = false
	freeze = false
	sleeping = false
	kick_velocity = velocity

func strike() -> void:
	flight = false
	freeze = false
	sleeping = false
	kick(-10.0)
	angular_velocity = Vector3.ZERO
