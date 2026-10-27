extends Node3D
## Rendering consumes public event frames; it never owns or predicts game rules.

const Card = preload("res://card.gd")
const Words = preload("res://words.gd")
const FONT = preload("res://assets/fonts/interface.tres")
const GOLD = Color("e9ca85")
const INK = Color("241a22")
const CREAM = Color("fff3d5")
const FLAT = Vector3(-PI / 2.0, 0, 0)
const BACK_FLAT = Vector3(PI / 2.0, 0, 0)
const STACK = Vector3(4.2, 0.0, -0.15)
const WorldOverlay = preload("res://world_overlay.gd")
const CARD_LAYER = 0.018
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
const JP_MONTHS = ["松", "梅", "桜", "藤", "菖蒲", "牡丹", "萩", "芒", "菊", "紅葉", "柳", "桐"]
var textures: Array[Texture2D] = []
var bodies: Array = []
var stack: Array = []
var known: Dictionary = {}
var opponent_cards: Dictionary = {}
var field_slots: Dictionary = {}
var camera: Camera3D
var request: HTTPRequest
var revision := 0
var busy := false
var started := false
var paused := false
var finished := false
var state: Dictionary = {}
var ending: Dictionary = {}
var language := 0
var speed := 1.0
var sound := true
var opponent := "trained"
var flags := 0
var menu: Node3D
var menu_buttons: Array = []
var label_nodes: Array = []
var hud: CanvasLayer
var left_label: Label3D
var right_label: Label3D
var bottom_label: Label3D
var overlay_world: Node3D
var stock_visual: Node3D
var pending_flights: Array = []
var role_order: Array[Array] = [[], []]
var role_slots: Array[Dictionary] = [{}, {}]
var capture_slots: Array[Dictionary] = [{}, {}]
var overlay: PanelContainer
var overlay_box: VBoxContainer
var modal := false
var audio: AudioStreamPlayer
var autoplay := false
var qa := false
var qa_frames: Dictionary = {}
var elapsed_match := 0.0
var testing := false

func word(key: String) -> String:
	return Words.get_text(key, language)

func _ready() -> void:
	autoplay = "--autoplay" in OS.get_cmdline_user_args()
	qa = "--qa" in OS.get_cmdline_user_args()
	testing = testing or autoplay or "--ui-test" in OS.get_cmdline_user_args()
	if not testing:
		load_settings()
	if autoplay:
		speed = 12.0
		opponent = "greedy"
	load_cards()
	make_world()
	make_hud()
	request = HTTPRequest.new()
	request.timeout = 45.0
	add_child(request)
	request.request_completed.connect(receive)
	audio = AudioStreamPlayer.new()
	audio.volume_db = -16
	add_child(audio)
	show_title()
	if qa:
		await get_tree().create_timer(1.0).timeout
		await screenshot("title")
	if autoplay:
		send("/start", {"opponent": opponent, "flags": flags})

func load_cards() -> void:
	for month in range(12):
		var kinds: Array = ["Tane", "Tanzaku", "Kasu_1", "Kasu_2"]
		if month in [0, 2]:
			kinds[0] = "Hikari"
		elif month == 7:
			kinds = ["Hikari", "Tane", "Kasu_1", "Kasu_2"]
		elif month == 10:
			kinds = ["Hikari", "Tane", "Tanzaku", "Kasu"]
		elif month == 11:
			kinds = ["Hikari", "Kasu_1", "Kasu_2", "Kasu_3"]
		for kind in kinds:
			textures.append(load("res://assets/cards/Hanafuda_%s_%s.png" % [MONTHS[month], kind]))

func mesh_box(size: Vector3, pos: Vector3, color: Color, parent: Node3D) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = size
	node.mesh = mesh
	node.material_override = Card.material(color)
	node.position = pos
	parent.add_child(node)
	return node

func label3(text: String, pos: Vector3, font_size: int = 42, color: Color = CREAM, parent: Node3D = null) -> Label3D:
	var label := Label3D.new()
	label.text = text
	label.font = FONT
	label.font_size = font_size
	label.pixel_size = 0.0085
	label.texture_filter = BaseMaterial3D.TEXTURE_FILTER_LINEAR_WITH_MIPMAPS_ANISOTROPIC
	label.outline_size = 0
	label.modulate = color
	label.position = pos
	label.no_depth_test = false
	label.double_sided = false
	if parent == null:
		add_child(label)
	else:
		parent.add_child(label)
	return label

func plane(size: Vector2, pos: Vector3, color: Color, parent: Node3D, border: Color = Color.TRANSPARENT) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var mesh := QuadMesh.new()
	mesh.size = size
	node.mesh = mesh
	node.material_override = Card.material(color)
	node.material_override.cull_mode = BaseMaterial3D.CULL_DISABLED
	node.position = pos
	parent.add_child(node)
	if border.a > 0:
		var outline := MeshInstance3D.new()
		var lines := ImmediateMesh.new()
		lines.surface_begin(Mesh.PRIMITIVE_LINE_STRIP)
		for uv in [Vector2(-1,-1), Vector2(1,-1), Vector2(1,1), Vector2(-1,1), Vector2(-1,-1)]:
			lines.surface_add_vertex(Vector3(uv.x * size.x / 2, uv.y * size.y / 2, 0.001))
		lines.surface_end()
		outline.mesh = lines
		outline.material_override = Card.material(border)
		node.add_child(outline)
	return node

func make_world() -> void:
	var environment := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color("b79232")
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	environment.environment = env
	add_child(environment)
	camera = Camera3D.new()
	camera.position = Vector3(0, 8.6, 10.6)
	camera.projection = Camera3D.PROJECTION_PERSPECTIVE
	camera.fov = 44.0
	add_child(camera)
	camera.look_at(Vector3(0, 1.8, 0))
	camera.current = true
	var board := plane(Vector2(10.6, 5.7), Vector3.ZERO, Color("4f063d"), self, Color("f4e7ba"))
	board.rotation.x = -PI / 2.0
	var table_body := StaticBody3D.new()
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(10.6, 0.01, 5.7)
	box.margin = 0.001
	shape.shape = box
	table_body.position.y = -0.005
	table_body.collision_layer = 1
	table_body.add_child(shape)
	add_child(table_body)
	stock_visual = Node3D.new()
	stock_visual.position = STACK
	add_child(stock_visual)
	for i in range(48):
		var card = Card.new()
		add_child(card)
		card.position = STACK + Vector3(0, i * CARD_LAYER, 0)
		card.rotation = BACK_FLAT
		card.stow()
		bodies.append(card)
		stack.append(card)
	update_stock()

func update_stock() -> void:
	for child in stock_visual.get_children():
		child.queue_free()
	if stack.is_empty():
		return
	var height := stack.size() * CARD_LAYER
	mesh_box(Vector3(Card.SIZE.x, height, Card.SIZE.y), Vector3(0, height / 2.0, 0), Color("b44519"), stock_visual)
	var outline := MeshInstance3D.new()
	var lines := ImmediateMesh.new()
	lines.surface_begin(Mesh.PRIMITIVE_LINES)
	var corners: Array[Vector3] = []
	for z in [-1, 1]:
		for y in [0, 1]:
			for x in [-1, 1]:
				corners.append(Vector3(x * Card.SIZE.x / 2, y * height, z * Card.SIZE.y / 2))
	for pair in [[0,1],[0,2],[1,3],[2,3],[4,5],[4,6],[5,7],[6,7],[0,4],[1,5],[2,6],[3,7]]:
		lines.surface_add_vertex(corners[pair[0]])
		lines.surface_add_vertex(corners[pair[1]])
	lines.surface_end()
	outline.mesh = lines
	outline.material_override = Card.material(Color("713d21"))
	stock_visual.add_child(outline)

func style(bg: Color, border: Color = Color.TRANSPARENT, radius: int = 4) -> StyleBoxFlat:
	var s := StyleBoxFlat.new()
	s.bg_color = bg
	s.border_color = border
	s.set_border_width_all(1)
	s.set_corner_radius_all(radius)
	s.content_margin_left = 20
	s.content_margin_right = 20
	s.content_margin_top = 12
	s.content_margin_bottom = 12
	return s

func ui_label(text: String, size: int, color: Color = CREAM) -> Label:
	var label := Label.new()
	label.text = text
	label.add_theme_font_override("font", FONT)
	label.add_theme_font_size_override("font_size", size)
	label.add_theme_color_override("font_color", color)
	return label

func ui_button(text: String, callback: Callable) -> Button:
	var b := Button.new()
	b.text = text
	b.add_theme_font_override("font", FONT)
	b.add_theme_font_size_override("font_size", 16)
	b.add_theme_color_override("font_color", CREAM)
	b.add_theme_stylebox_override("normal", style(INK, Color("806b48")))
	b.add_theme_stylebox_override("hover", style(Color("513c37"), GOLD))
	b.add_theme_stylebox_override("pressed", style(Color("6b433d"), GOLD))
	b.pressed.connect(callback)
	return b

func make_hud() -> void:
	# Round, scores and turn status inhabit the same upright world plane as decisions.
	left_label = label3("", Vector3(-5.65, 2.0, 0.0), 26, CREAM)
	right_label = label3("", Vector3(5.65, 2.0, 0.0), 26, CREAM)
	bottom_label = label3("", Vector3(0, 1.32, 0.0), 25, GOLD)
	hud = CanvasLayer.new()
	add_child(hud)
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud.add_child(root)
	var settings := ui_button("☰", func(): open_overlay("settings"))
	settings.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_RIGHT)
	settings.position = Vector2(-78, -66)
	settings.size = Vector2(50, 42)
	root.add_child(settings)
	refresh_hud()

func clear_menu() -> void:
	if is_instance_valid(menu):
		menu.queue_free()
	menu_buttons.clear()
	menu = Node3D.new()
	menu.position = Vector3(0, 2.7, 0)
	add_child(menu)

func panel(title: String, subtitle: String = "", reverse: String = "", _height: float = 0.9, mirrored: bool = false) -> void:
	clear_menu()
	var width := clampf(FONT.get_string_size(title, HORIZONTAL_ALIGNMENT_LEFT, -1, 46).x * 0.0085 + 0.65, 3.7, 7.2)
	plane(Vector2(width, 0.92), Vector3.ZERO, Color("3d301c"), menu, Color("f2bc58"))
	var text := label3(title, Vector3(0, 0, 0.006), 46, Color("c73c0b"), menu)
	text.outline_size = 4
	text.outline_modulate = Color("ffce65")
	if not subtitle.is_empty():
		label3(subtitle, Vector3(0, -0.75, 0.006), 25, CREAM, menu)
	var back := label3(reverse if reverse != "" else title, Vector3(0, 0, -0.006), 46, Color("c73c0b"), menu)
	back.outline_size = 4
	back.outline_modulate = Color("ffce65")
	back.rotation.y = PI
	if mirrored:
		back.scale.x = -1.0

func world_button(text: String, y: float, action: String, value: int = 0, x: float = 0, width: float = 3.4) -> void:
	plane(Vector2(width, 0.48), Vector3(x, y, 0.008), Color("49351e"), menu, GOLD)
	label3(text, Vector3(x, y, 0.013), 29, CREAM, menu)
	menu_buttons.append({"x": x, "y": y, "width": width, "height": 0.48, "action": action, "value": value})

func show_title() -> void:
	panel("START")
	menu_buttons.append({"x": 0.0, "y": 0.0, "width": 3.7, "height": 0.92, "action": "start", "value": 0})
	refresh_hud()

func switch_language() -> void:
	language = 1 - language
	save_settings()
	refresh_hud()
	if not busy:
		show_decision()
	if modal:
		open_overlay("settings")

func refresh_hud() -> void:
	if state.is_empty():
		left_label.text = ""
		right_label.text = ""
		bottom_label.text = ""
		return
	bottom_label.visible = not is_instance_valid(menu) or menu.get_child_count() == 0
	var pts: Array = state.points
	left_label.text = "%s    %02d / 12\n%s  %d %s%s" % [word("round"), state.round, word("you"), pts[0], word("points"), "  · " + word("dealer") if state.dealer == 0 else ""]
	right_label.text = "%s  %d %s%s\n%s" % [word("ai"), pts[1], word("points"), "  · " + word("dealer") if state.dealer == 1 else "", word("turn_you") if state.player == 0 else word("turn_ai")]
	bottom_label.text = word("thinking") if busy or state.player == 1 else word("pick") if state.phase in [1, 2] else word("play")
	if paused:
		bottom_label.text = word("final") if finished else "%s %d / 12" % [word("round"), state.round]
		right_label.text = right_label.text.get_slice("\n", 0) + "\n" + word("final" if finished else "round_complete")

func send(path: String, data: Dictionary = {}) -> void:
	if busy:
		return
	busy = true
	data["revision"] = revision
	if path == "/start":
		panel(word("loading"))
	var url := OS.get_environment("KOIKOI_URL")
	var token := OS.get_environment("KOIKOI_TOKEN")
	if url.is_empty():
		show_error("play.cmd から起動してください。 / Launch with play.cmd.")
		return
	var err := request.request(url + path, ["Content-Type: application/json", "Authorization: Bearer " + token], HTTPClient.METHOD_POST, JSON.stringify(data))
	if err != OK:
		show_error("Request could not start: %s" % err)
	refresh_hud()

func receive(result: int, code: int, _headers: PackedStringArray, body: PackedByteArray) -> void:
	var parsed = JSON.parse_string(body.get_string_from_utf8())
	if result != HTTPRequest.RESULT_SUCCESS or not parsed is Dictionary:
		show_error("The local game service did not respond. / ローカルサービスに接続できません。")
		return
	revision = int(parsed.get("revision", revision))
	if code != 200:
		show_error(str(parsed.get("error", "Unknown error")))
		return
	paused = parsed.get("paused", false)
	finished = parsed.get("finished", false)
	ending = parsed.get("end", {}) if parsed.get("end") != null else {}
	clear_menu()
	for event in parsed.get("events", []):
		await animate_event(event)
	state = parsed.get("view", {})
	if not state.is_empty():
		started = true
		if not paused:
			await arrange_captures(state)
	busy = false
	show_decision()
	refresh_hud()
	if not state.is_empty():
		await screenshot("round_end" if paused else "koi" if state.phase == 3 else "table")
	if autoplay and finished:
		print("GAME_QA_COMPLETE rounds=12 points=%s bodies=%d" % [state.points, bodies.size()])
		get_tree().quit(0)
		return
	await maybe_ai()

func show_error(message: String) -> void:
	busy = false
	panel(word("error"), message.left(180), "", 4.0)
	world_button(word("retry"), -1.1, "retry")
	push_error(message)

func show_decision() -> void:
	if not started:
		show_title()
	elif paused:
		show_result()
	elif int(state.phase) == 3:
		panel(word("koi_question"), "%d %s" % [state.yaku_points[int(state.player)], word("points")], word("koi_question"), 3.3, true)
		menu.position.y = 4.9
		if int(state.player) == 0:
			world_button(word("yes"), -1.35, "action", 49, -1.4, 2.5)
			world_button(word("no"), -1.35, "action", 48, 1.4, 2.5)
		else:
			menu.rotation.y = PI
	else:
		clear_menu()

func show_result() -> void:
	var winner := int(ending.get("winner", 0))
	var headline := "WIN" if winner == 0 else "LOSE"
	if finished:
		headline = "DRAW" if state.points[0] == state.points[1] else "WIN" if state.points[0] > state.points[1] else "LOSE"
	var rows: Array = state.yaku[winner]
	var text := "%s 12 · %s\n" % [word("round"), word("you" if winner == 0 else "ai")] if finished else ""
	if ending.get("reason", "stop") == "exhausted":
		rows = []
	for row in rows:
		text += "%s  %d\n" % [word(row.key), row.points]
	if ending.get("total_yaku", false) and not state.yaku[1 - winner].is_empty():
		for row in state.yaku[1 - winner]:
			text += "%s · %s  %d\n" % [word("ai" if winner == 0 else "you"), word(row.key), row.points]
	if rows.is_empty():
		text += word("exhausted") + "\n"
	var base_points := int(state.yaku_points[winner]) + (int(state.yaku_points[1 - winner]) if ending.get("total_yaku", false) else 0)
	if base_points > 0 and int(ending.get("payment", 0)) == base_points * 2:
		text += "KOI KOI × 2\n"
	text += "\n+%d %s" % [ending.get("payment", 0), word("points")]
	var lines := text.split("\n")
	var columns := lines.size() > 8
	var height := maxf(1.1, (ceilf(lines.size() / 2.0) if columns else lines.size()) * 0.29 + 0.45)
	panel(headline, word("final") if finished else "%s %d / 12" % [word("round"), state.round], "LOSE" if headline == "WIN" else "WIN" if headline == "LOSE" else "DRAW", height)
	menu.get_child(0).material_override = Card.material(Color("b94013"))
	for child in menu.get_children():
		if child is Label3D and child.font_size == 46:
			child.modulate = CREAM
			child.outline_modulate = Color("18797d")
	menu.position.y = 3.8 + maxf(0, height - 1.6) * 0.5
	plane(Vector2(6.6, height), Vector3(0, -0.92 - height / 2.0, 0.002), Color("3d301c"), menu, GOLD)
	if columns:
		var middle := ceili(lines.size() / 2.0)
		label3("\n".join(lines.slice(0, middle)), Vector3(-1.65, -0.92 - height / 2.0, 0.006), 23, CREAM, menu)
		label3("\n".join(lines.slice(middle)), Vector3(1.65, -0.92 - height / 2.0, 0.006), 23, CREAM, menu)
	else:
		label3(text, Vector3(0, -0.92 - height / 2.0, 0.006), 25, CREAM, menu)
	world_button(word("new" if finished else "next"), -1.28 - height, "start" if finished else "next")

func maybe_ai() -> void:
	if busy or modal or state.is_empty() or finished:
		return
	if autoplay and paused:
		send("/next")
	elif not paused and int(state.player) == 1:
		await wait_for(0.55)
		if not busy and not modal:
			send("/ai")
	elif autoplay and not paused:
		var actions: Array = state.legal
		if not actions.is_empty():
			send("/action", {"action": int(actions[0])})

func wait_for(seconds: float) -> void:
	await get_tree().create_timer(maxf(0.018, seconds / speed)).timeout

func take_card(id: int = -1):
	var card = stack.pop_back()
	card.position = STACK + Vector3(0, (stack.size() + 1) * CARD_LAYER + 0.006, 0)
	card.rotation = BACK_FLAT
	card.identity = id
	card.reveal_turn = false
	card.pair_peer = null
	card.set_face(textures[id] if id >= 0 else null)
	card.unstow()
	if id >= 0:
		known[id] = card
	update_stock()
	return card

func hand_position(slot: int, player: int) -> Vector3:
	return Vector3((slot - 3.5) * 0.99, 1.04, 3.9 if player == 0 else -4.25)

func field_position(card_id: int) -> Vector3:
	if not field_slots.has(card_id):
		var occupied := field_slots.values()
		for i in range(32):
			if i not in occupied:
				field_slots[card_id] = i
				break
	var index: int = field_slots[card_id]
	if index < 8:
		return Vector3((index % 4 - 1.5) * 1.32 - 0.9, 0.012, (index / 4 - 0.5) * 1.7)
	if index < 16:
		return Vector3((index % 4 - 1.5) * 1.32 - 0.9, 0.012, -2.18 if index < 12 else 2.18)
	return Vector3(-4.25 if index < 20 else 2.5, 0.012 + maxf(0, index - 24) * 0.018, ((index - 16) % 4 - 1.5) * 1.4)

func half_bounds(basis: Basis) -> Vector3:
	return basis.x.abs() * Card.SIZE.x / 2 + basis.y.abs() * Card.SIZE.y / 2 + basis.z.abs() * Card.SIZE.z / 2

func fly(card, pos: Vector3, rotation: Vector3, seconds: float = 0.55, requested_lift: float = 0.0, circle: bool = false) -> void:
	card.move_to(pos, rotation, seconds / speed, requested_lift, circle)
	pending_flights.append(card)

func flight_root(card):
	var leader = card
	while leader.pair_peer != null:
		leader = leader.pair_peer
	return leader

func plan_flights() -> void:
	var flights := pending_flights.duplicate()
	pending_flights.clear()
	# Register every destination before considering the other moving cards.
	for card in flights:
		for other in flights:
			if other == card or card.pair_peer != null or other.pair_peer == card:
				continue
			if card.flight_from.y >= other.flight_from.y:
				continue
			if absf(card.duration - other.duration) > 0.00001:
				continue
			if absf(card.rotation_from.dot(other.rotation_from)) < 0.999 or absf(card.rotation_to.dot(other.rotation_to)) < 0.999:
				continue
			var relative: Vector3 = Basis(card.rotation_from).inverse() * (card.flight_from - other.flight_from)
			if absf(relative.x) < Card.SIZE.x and absf(relative.y) < Card.SIZE.y and absf(relative.z) < 0.1:
				card.pair_peer = other
				card.pair_sign = 1.0 if relative.z >= 0 else -1.0
	flights.sort_custom(func(a, b):
		var ar = flight_root(a)
		var br = flight_root(b)
		if ar == br:
			return a.flight_from.y > b.flight_from.y
		if ar.flight_from.y != br.flight_from.y:
			return ar.flight_from.y > br.flight_from.y
		return ar.get_instance_id() < br.get_instance_id())
	for i in range(flights.size()):
		var card = flights[i]
		plan_clearance(card, flights.slice(i + 1))
		# Finish a paired group's shared clearance before planning later flights.
		if card.pair_peer != null:
			var peer = card.pair_peer
			for sample in range(1, 96):
				var height: float = maxf(card.clearance[sample], peer.clearance[sample])
				card.clearance[sample] = height
				peer.clearance[sample] = height

func plan_clearance(card, lower_priority: Array) -> void:
	# Minimum clearance above the board and the current/future card footprints.
	# Position and orientation are solved together; no arbitrary central waypoint.
	for iteration in range(12):
		var changed := false
		for i in range(1, 96):
			var t := i / 96.0
			var pose: Transform3D = card.pose_at(t * card.duration)
			var extent := half_bounds(pose.basis)
			var floor_y := -100.0
			if absf(pose.origin.x) < 5.3 + extent.x and absf(pose.origin.z) < 2.85 + extent.z:
				floor_y = 0.006
			# The solid stock participates as one box, never 48 exposed slabs.
			if absf(pose.origin.x - STACK.x) < extent.x + Card.SIZE.x / 2 and absf(pose.origin.z - STACK.z) < extent.z + Card.SIZE.y / 2:
				floor_y = maxf(floor_y, stack.size() * CARD_LAYER + 0.006)
			for other in bodies:
				if other == card or other in lower_priority or not other.visible or card.pair_peer == other or other.pair_peer == card:
					continue
				var other_pose: Transform3D = other.pose_at(other.elapsed + t * card.duration) if other.flight else other.transform
				var other_extent := half_bounds(other_pose.basis)
				var diff := (pose.origin - other_pose.origin).abs()
				if diff.x < extent.x + other_extent.x + 0.03 and diff.z < extent.z + other_extent.z + 0.03:
					# Don't route above upright racks that are entirely above this route.
					if other.flight or pose.origin.y + extent.y >= other_pose.origin.y - other_extent.y - 0.01:
						floor_y = maxf(floor_y, other_pose.origin.y + other_extent.y + 0.025)
			var deficit := floor_y + extent.y - pose.origin.y
			if deficit > 0:
				card.clearance[i] += deficit + 0.003
				changed = true
		var expanded = card.clearance.duplicate()
		for i in range(1, 96):
			for j in range(maxi(1, i - 2), mini(96, i + 3)):
				expanded[j] = maxf(expanded[j], card.clearance[i])
		card.clearance = expanded
		# Limit sharp changes while retaining the minimum local obstacle clearance.
		for i in range(94, 0, -1):
			card.clearance[i] = maxf(card.clearance[i], card.clearance[i + 1] - 0.12)
		for i in range(2, 96):
			card.clearance[i] = maxf(card.clearance[i], card.clearance[i - 1] - 0.12)
		if not changed:
			break

func finish_flights() -> void:
	plan_flights()
	while bodies.any(func(c): return c.flight):
		await get_tree().physics_frame

func reset_deck() -> void:
	clear_yaku_labels()
	known.clear()
	opponent_cards.clear()
	field_slots.clear()
	capture_slots = [{}, {}]
	role_order = [[], []]
	role_slots = [{}, {}]
	pending_flights.clear()
	# A short ordered return peels exposed cards off first, avoiding crossing lanes.
	var returning: Array = bodies.filter(func(c): return c.visible)
	returning.sort_custom(func(a, b): return a.position.y > b.position.y)
	for card in returning:
		card.pair_peer = null
		card.reveal_turn = false
		fly(card, STACK + Vector3(0, (stack.size() + 1) * CARD_LAYER, 0), BACK_FLAT, 0.24)
		await finish_flights()
		card.stow()
		card.identity = -1
		card.set_face(null)
		stack.append(card)
		update_stock()

func opening(event: Dictionary) -> void:
	await reset_deck()
	for reveal in event.reveals:
		panel(word("opening"))
		menu.position.y = 4.4
		var pair: Array = []
		for p in range(2):
			var card = take_card(int(reveal.cards[p]))
			pair.append(card)
			fly(card, Vector3(-0.7 if p == 0 else 0.7, 2.25, 0), Vector3(0, PI * p, 0), 0.65)
		await finish_flights()
		await wait_for(0.3)
		# Dealer reveal is the one explicitly requested stationary turn.
		for card in pair:
			card.move_to(card.position, card.rotation + Vector3(0, PI, 0), 1.1 / speed)
		await finish_flights()
		panel(word("tie") if int(reveal.winner) < 0 else word("dealer_you" if int(reveal.winner) == 0 else "dealer_ai"))
		menu.position.y = 4.4
		await wait_for(0.55)
		await reset_deck()
	clear_menu()
	await deal_cards(event.view, false)

func deal_cards(view: Dictionary, returning: bool = true) -> void:
	if returning:
		await reset_deck()
	clear_menu()
	state = view
	refresh_hud()
	for batch in range(4):
		for group in [2, int(view.dealer), 1 - int(view.dealer)]:
			var pair: Array = []
			var destinations: Array[Vector3] = []
			for j in range(2):
				var slot := batch * 2 + j
				var id := int(view.field[slot]) if group == 2 else int(view.hand[slot]) if group == 0 else -1
				var card = take_card(id)
				pair.append(card)
				if group == 2:
					destinations.append(field_position(id))
				else:
					card.slot = slot
					card.seat = group
					if group == 1:
						opponent_cards[slot] = card
					destinations.append(hand_position(slot, group))
			pair[1].pair_peer = pair[0]
			# Both flights start within this frame, with the same duration/rotation.
			for j in range(2):
				fly(pair[j], destinations[j], FLAT if group == 2 else Vector3(0, PI * group, 0), 0.6)
			await finish_flights()
			tap(0.65)

func needs_pick(view: Dictionary, card_id: int) -> bool:
	return view.field.filter(func(c): return int(c) / 4 == card_id / 4).size() == 2

func animate_event(event: Dictionary) -> void:
	var kind: String = event.kind
	var view: Dictionary = event.view
	if kind == "opening":
		await opening(event)
	elif kind == "deal":
		await deal_cards(view)
	elif kind == "sync":
		await restore_view(view)
	elif kind == "play":
		var id := int(event.card)
		var card
		if int(event.player) == 1:
			card = opponent_cards[int(event.slot)]
			opponent_cards.erase(int(event.slot))
			card.identity = id
			card.set_face(textures[id])
			known[id] = card
		else:
			card = known[id]
		card.reveal_turn = true
		if needs_pick(view, id):
			fly(card, Vector3(2.65, 0.1, 0), FLAT, 0.62)
			await finish_flights()
	elif kind == "draw":
		var id := int(event.card)
		var card = take_card(id)
		card.rotation = FLAT # The specified exception: instantaneous stock-top flip.
		if needs_pick(view, id):
			fly(card, Vector3(2.65, 0.1, 0), FLAT, 0.48)
			await finish_flights()
	elif kind in ["capture", "place"]:
		var card = known[int(event.card)]
		var target: Vector3 = field_position(int(event.card)) if kind == "place" else known[int(event.targets[0])].position + Vector3(0.13, 0.032, 0.13)
		fly(card, target + Vector3(0, 0.14 if event.new_yaku else 0, 0), FLAT, 0.62)
		await finish_flights()
		if event.new_yaku:
			card.strike()
			await get_tree().create_timer(0.24).timeout
		tap(1.2 if event.new_yaku else 0.8)
		if kind == "capture":
			for c in event.targets:
				field_slots.erase(int(c))
			await arrange_captures(view)
	elif kind == "koi":
		panel(word("koi_you" if int(event.player) == 0 else "koi_ai"), "", word("koi_you" if int(event.player) == 0 else "koi_ai"), 0.9, true)
		menu.position.y = 4.9
		if int(event.player) == 0:
			menu.rotation.y = PI
		await wait_for(0.9)
		clear_menu()
	elif kind == "round_end":
		if event.total_yaku:
			await sweep_yaku(view, int(event.winner))
			await screenshot("total_yaku")
	state = view
	refresh_hud()

func restore_view(view: Dictionary) -> void:
	# Recover public objects through visible, collision-aware flights.
	await reset_deck()
	for slot in range(8):
		var id := int(view.hand[slot])
		if id >= 0:
			var card = take_card(id)
			card.slot = slot
			card.seat = 0
			fly(card, hand_position(slot, 0), Vector3.ZERO, 0.3)
			await finish_flights()
	for raw_slot in view.opponent_slots:
		var slot := int(raw_slot)
		var card = take_card()
		card.slot = slot
		card.seat = 1
		opponent_cards[slot] = card
		fly(card, hand_position(slot, 1), Vector3(0, PI, 0), 0.3)
		await finish_flights()
	for raw_id in view.field:
		var id := int(raw_id)
		fly(take_card(id), field_position(id), FLAT, 0.3)
		await finish_flights()
	if int(view.pending) >= 0:
		fly(take_card(int(view.pending)), Vector3(2.65, 0.1, 0), FLAT, 0.3)
		await finish_flights()
	await arrange_captures(view, true)
	if paused and ending.get("total_yaku", false):
		await sweep_yaku(view, int(ending.winner))

func clear_yaku_labels() -> void:
	for node in label_nodes:
		node.queue_free()
	label_nodes.clear()

func role_key(key: String) -> String:
	return "lights" if key in ["three", "four", "rain_four", "five"] else key

func arrange_captures(view: Dictionary, serial: bool = false) -> void:
	clear_yaku_labels()
	var changed := false
	for player in range(2):
		var allocated: Array = []
		var row_index := 0
		var rows: Array = view.yaku[player].filter(func(row): return not row.cards.is_empty())
		for row in rows:
			var key := role_key(row.key)
			if key not in role_order[player]:
				role_order[player].append(key)
				role_slots[player][key] = {}
		rows.sort_custom(func(a, b): return role_order[player].find(role_key(a.key)) < role_order[player].find(role_key(b.key)))
		for row in rows:
			var slots: Dictionary = role_slots[player][role_key(row.key)]
			if row.cards.is_empty():
				continue
			var cards_here: Array = []
			for raw_id in row.cards:
				var id := int(raw_id)
				if id not in allocated:
					allocated.append(id)
					cards_here.append(id)
					if not slots.has(id):
						slots[id] = slots.size()
			var z := 9.3 if player == 0 else -9.3
			var y := 3.55 - row_index * 1.52
			var plaque := Node3D.new()
			plaque.position = Vector3(4.6 if player == 0 else -4.6, y, z)
			plaque.rotation.y = PI if player == 0 else 0.0
			add_child(plaque)
			plane(Vector2(1.9, 0.4), Vector3.ZERO, Color("3d301c"), plaque, GOLD)
			label3(word(row.key) + "  " + str(row.points), Vector3(0, 0, 0.006), 18, GOLD, plaque)
			label_nodes.append(plaque)
			for i in range(cards_here.size()):
				var card = known[cards_here[i]] if known.has(cards_here[i]) else take_card(cards_here[i])
				var slot: int = slots[cards_here[i]]
				var spacing := 0.88
				var x := (3.0 - (slot % 10) * spacing) * (1 if player == 0 else -1)
				var pos := Vector3(x, y - (slot / 10) * 1.52, z)
				# Real cards occupy just one role row; overlapping roles share membership.
				if card.position.distance_to(pos) > 0.015:
					fly(card, pos, Vector3(0, PI if player == 0 else 0.0, 0), 0.6)
					changed = true
					if serial:
						await finish_flights()
			row_index += maxi(1, ceili(slots.size() / 10.0))
		for id in allocated:
			capture_slots[player].erase(id)
		var index := 0
		for raw_id in view.captured[player]:
			var id := int(raw_id)
			if id in allocated:
				continue
			if not capture_slots[player].has(id):
				var used: Array = capture_slots[player].values()
				var free := 0
				while free in used:
					free += 1
				capture_slots[player][id] = free
			index = capture_slots[player][id]
			var pos := Vector3(-3.465 + (index % 8) * 0.99, 0.014, (3.18 + (index / 8) * 1.42) * (1 if player == 0 else -1))
			var card = known[id] if known.has(id) else take_card(id)
			if card.position.distance_to(pos) > 0.015:
				fly(card, pos, FLAT if player == 0 else Vector3(-PI / 2, 0, PI), 0.55)
				changed = true
				if serial:
					await finish_flights()
			index += 1
	if changed:
		await finish_flights()

func sweep_yaku(view: Dictionary, winner: int) -> void:
	clear_yaku_labels()
	var ids: Array = []
	for player in range(2):
		for row in view.yaku[player]:
			for c in row.cards:
				if int(c) not in ids:
					ids.append(int(c))
	for i in range(ids.size()):
		var target := Vector3((i % 12 - 5.5) * 0.95, 3.55 - (i / 12) * 1.52, 11.8 if winner == 0 else -11.8)
		fly(known[ids[i]], target, Vector3(0, PI if winner == 0 else 0, 0), 1.2, 0.0, true)

	await finish_flights()

func tap(intensity: float = 1.0) -> void:
	if not sound or autoplay:
		return
	var sample := AudioStreamWAV.new()
	sample.format = AudioStreamWAV.FORMAT_16_BITS
	sample.mix_rate = 22050
	var bytes := PackedByteArray()
	bytes.resize(2205 * 2)
	for i in range(2205):
		var t := float(i) / 22050.0
		var value := (sin(t * 1950) + sin(t * 3270) * 0.4) * exp(-t * 65) * 5600 * intensity
		bytes.encode_s16(i * 2, int(value))
	sample.data = bytes
	audio.stream = sample
	audio.play()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_F11:
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED if DisplayServer.window_get_mode() == DisplayServer.WINDOW_MODE_FULLSCREEN else DisplayServer.WINDOW_MODE_FULLSCREEN)
		elif event.keycode == KEY_ESCAPE:
			if modal:
				close_overlay()
			else:
				open_overlay("settings")
	if modal:
		overlay_world.route(event, camera)
		return
	if busy:
		return
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		if click_menu(event.position):
			return
		if started and not paused and int(state.player) == 0:
			var card = hit_card(event.position)
			if card != null and state.legal.any(func(a): return int(a) == card.identity):
				send("/action", {"action": card.identity})

func hit_card(screen: Vector2):
	var ray := PhysicsRayQueryParameters3D.create(camera.project_ray_origin(screen), camera.project_ray_origin(screen) + camera.project_ray_normal(screen) * 80.0, 2)
	var hit := get_world_3d().direct_space_state.intersect_ray(ray)
	return hit.get("collider")

func click_menu(screen: Vector2) -> bool:
	if not is_instance_valid(menu) or menu_buttons.is_empty():
		return false
	var ray := camera.project_ray_normal(screen)
	var origin := camera.project_ray_origin(screen)
	var plane := Plane(menu.global_basis.z, menu.global_position + menu.global_basis.z * 0.013)
	var point = plane.intersects_ray(origin, ray)
	if point == null:
		return false
	var local: Vector3 = menu.to_local(point)
	for item in menu_buttons:
		if absf(local.x - item.x) <= item.width / 2.0 and absf(local.y - item.y) <= item.height / 2.0:
			match item.action:
				"start": send("/start", {"opponent": opponent, "flags": flags})
				"action": send("/action", {"action": item.value})
				"next": send("/next")
				"settings": open_overlay("settings")
				"retry": send("/state")
			return true
	return false

func open_overlay(kind: String) -> void:
	modal = true
	if is_instance_valid(overlay_world):
		overlay_world.queue_free()
	overlay_world = WorldOverlay.new()
	overlay_world.position = Vector3(0, 3.3, 0.4)
	add_child(overlay_world)
	overlay = PanelContainer.new()
	overlay.add_theme_stylebox_override("panel", style(INK, GOLD, 3))
	overlay.size = Vector2(820, 780)
	overlay_world.viewport.add_child(overlay)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(780, 740)
	overlay.add_child(scroll)
	overlay_box = VBoxContainer.new()
	overlay_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	overlay_box.add_theme_constant_override("separation", 13)
	scroll.add_child(overlay_box)
	var top := HBoxContainer.new()
	overlay_box.add_child(top)
	var title := ui_label(word(kind), 27, GOLD)
	title.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	top.add_child(title)
	top.add_child(ui_button(word("close"), close_overlay))
	if kind == "help":
		var help := ui_label(word("help_body"), 16)
		help.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		overlay_box.add_child(help)
	elif kind == "credits":
		var credits := ui_label("HANAFUDA\nLouie Mantia · すけじょ (traditional colors)\nCC BY-SA 4.0\n\ncommons.wikimedia.org\nFrames recolored to #b44519; illustration ink retained.\n\nNoto Sans JP · The Noto Project\nSIL Open Font License 1.1\n\nGodot Engine 4.7.2 · MIT\ngodotengine.org\n\nRules & AI integration\nKoiKoi-AI · Sanghai Guan · MIT\n\nFull notices: game/assets and THIRD_PARTY_NOTICES.md", 17)
		overlay_box.add_child(credits)
	else:
		overlay_box.add_child(ui_button("日本語 / ENGLISH", switch_language))
		overlay_box.add_child(ui_label(word("opponent"), 17, GOLD))
		var options := OptionButton.new()
		options.add_theme_font_override("font", FONT)
		options.add_theme_font_size_override("font_size", 17)
		var names := ["trained", "rl-wp", "rl-point", "greedy"]
		for name in names:
			options.add_item(word(name) if name in ["trained", "greedy"] else name.to_upper())
		options.select(names.find(opponent))
		options.item_selected.connect(func(i): opponent = names[i]; save_settings())
		overlay_box.add_child(options)
		overlay_box.add_child(ui_label(word("rules") + " · 12", 17, GOLD))
		for rule in [{"key": "total", "bit": 16}, {"key": "double", "bit": 8}]:
			var toggle := CheckButton.new()
			toggle.text = word(rule.key)
			toggle.add_theme_font_override("font", FONT)
			toggle.add_theme_font_size_override("font_size", 17)
			toggle.button_pressed = (flags & int(rule.bit)) != 0
			toggle.toggled.connect(func(on): flags = flags | int(rule.bit) if on else flags & ~int(rule.bit); save_settings())
			overlay_box.add_child(toggle)
		overlay_box.add_child(ui_label(word("apply_next"), 14, GOLD))
		overlay_box.add_child(ui_label(word("speed"), 17))
		var slider := HSlider.new()
		slider.min_value = 0.6
		slider.max_value = 2.0
		slider.step = 0.1
		slider.value = speed
		slider.value_changed.connect(func(v): speed = v; save_settings())
		overlay_box.add_child(slider)
		var sound_toggle := CheckButton.new()
		sound_toggle.text = word("sound")
		sound_toggle.add_theme_font_override("font", FONT)
		sound_toggle.button_pressed = sound
		sound_toggle.toggled.connect(func(v): sound = v; save_settings())
		overlay_box.add_child(sound_toggle)
		overlay_box.add_child(ui_button(word("help"), func(): open_overlay("help")))
		overlay_box.add_child(ui_button(word("credits"), func(): open_overlay("credits")))

func close_overlay() -> void:
	if is_instance_valid(overlay_world):
		overlay_world.queue_free()
	modal = false
	await maybe_ai()

func load_settings() -> void:
	var config := ConfigFile.new()
	if config.load("user://settings.cfg") == OK:
		language = clampi(int(config.get_value("game", "language", 0)), 0, 1)
		speed = clampf(float(config.get_value("game", "speed", 1.0)), 0.6, 2.0)
		sound = bool(config.get_value("game", "sound", true))
		opponent = str(config.get_value("game", "opponent", "trained"))
		flags = int(config.get_value("game", "flags", 0)) & 24

func save_settings() -> void:
	if testing:
		return
	var config := ConfigFile.new()
	for pair in [["language", language], ["speed", speed], ["sound", sound], ["opponent", opponent], ["flags", flags]]:
		config.set_value("game", pair[0], pair[1])
	config.save("user://settings.cfg")

func screenshot(name: String) -> void:
	if not qa or qa_frames.has(name) or DisplayServer.get_name() == "headless":
		return
	qa_frames[name] = true
	await RenderingServer.frame_post_draw
	var path := ProjectSettings.globalize_path("res://../output/game-qa")
	DirAccess.make_dir_recursive_absolute(path)
	get_viewport().get_texture().get_image().save_png(path.path_join(name + ".png"))

func _process(delta: float) -> void:
	if autoplay:
		elapsed_match += delta
		if elapsed_match > 300:
			push_error("Autoplay watchdog expired")
			get_tree().quit(2)
