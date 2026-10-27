extends Node3D
## Rendering consumes public event frames; it never owns or predicts game rules.

const Card = preload("res://card.gd")
const Words = preload("res://words.gd")
const SERIF = preload("res://assets/fonts/NotoSerifJP-Black.otf")
const Layout = preload("res://layout.gd")
var latin_font: FontFile
var FONT: Font:
	get: return SERIF if language == 0 else latin_font
var latin_font_loaded := false
const GOLD = Color("e9ca85")
const INK = Color("241a22")
const CREAM = Color("fff3d5")
const FLAT = Vector3(-PI / 2.0, 0, 0)
const BACK_FLAT = Vector3(PI / 2.0, 0, 0)
const STACK = Vector3(4.2, 0.0, -0.15)
const WorldOverlay = preload("res://world_overlay.gd")
const CARD_LAYER = 0.018
const SOUNDS = {
	"tefuda-place": preload("res://assets/audio/tefuda-place.mp3"),
	"bafuda-place": preload("res://assets/audio/bafuda-place.mp3"),
	"bafuda-place-yaku": preload("res://assets/audio/bafuda-place-yaku.mp3"),
	"fuda-take": preload("res://assets/audio/fuda-take.mp3"),
	"fuda-turn": preload("res://assets/audio/fuda-turn.mp3"),
	"yaku": preload("res://assets/audio/yaku.mp3"),
	"ui-click": preload("res://assets/audio/ui-click.wav"),
	"ui-message": preload("res://assets/audio/ui-message.wav"),
	"win": preload("res://assets/audio/win.wav"),
	"lose": preload("res://assets/audio/lose.wav"),
}
# UI tones are balanced to about -30 dBFS RMS before the shared master gain,
# matching the hand/draw clips rather than their much higher original tone RMS.
const SOUND_GAIN_DB = {"ui-click": -9.4, "ui-message": -10.0, "win": -10.9, "lose": -11.1}

var sound_log: Array = []
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
const JP_MONTHS = ["松", "梅", "桜", "藤", "菖蒲", "牡丹", "萩", "芒", "菊", "紅葉", "柳", "桐"]
var textures: Array[Texture2D] = []
var bodies: Array = []
var stack: Array = []
var known: Dictionary = {}
var opponent_cards: Dictionary = {}
var field_slots: Dictionary = {}
var field_columns := 4
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
var match_rounds := 12
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
var autoplay := false
var qa := false
var qa_frames: Dictionary = {}
var elapsed_match := 0.0
var testing := false
var board: MeshInstance3D
var board_body: StaticBody3D
var board_shape: BoxShape3D
var board_left := -5.3
var selected_hand := -1
var queued_capture := -1
var active_pending := -1
var pending_origin := Vector3.ZERO
var pending_angles := Vector3.ZERO
var announcement: Label3D
var announcement_history: Array[String] = []
var drag_start := Vector2.ZERO
var camera_drag := false
var drag_distance := 0.0
var orbit_pitch := atan2(6.8, 10.6)
var camera_radius := Vector2(6.8, 10.6).length()
var camera_pivot := Vector3(0, 1.8, 0)
var returning_all := false
var last_burst_peak := 0.0
var decision_announced_revision := -1
var last_impact_settled := false

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
	load_fonts()
	load_cards()
	make_world()
	make_hud()
	request = HTTPRequest.new()
	request.timeout = 45.0
	add_child(request)
	request.request_completed.connect(receive)
	show_title()
	if qa:
		await get_tree().create_timer(1.0).timeout
		await screenshot("title")
	if autoplay:
		send("/start", {"opponent": opponent, "flags": flags})

func load_fonts() -> void:
	var latin: FontFile = preload("res://assets/fonts/AntiqueSix-ExtraBold.ttf").duplicate()
	latin.fallbacks = [SERIF]
	latin_font = latin
	latin_font_loaded = true

func update_camera() -> void:
	camera.position = camera_pivot + Vector3(0, sin(orbit_pitch), cos(orbit_pitch)) * camera_radius
	camera.look_at(camera_pivot)

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
	label.alpha_cut = Label3D.ALPHA_CUT_OPAQUE_PREPASS
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
	board = plane(Vector2(10.6, 5.7), Vector3.ZERO, Color("4f063d"), self, Color("f4e7ba"))
	board.rotation.x = -PI / 2.0
	board_body = StaticBody3D.new()
	var shape := CollisionShape3D.new()
	board_shape = BoxShape3D.new()
	board_shape.size = Vector3(10.6, 0.8, 5.7)
	board_shape.margin = 0.001
	shape.shape = board_shape
	board_body.position.y = -0.4
	board_body.collision_layer = 1
	board_body.add_child(shape)
	add_child(board_body)
	stock_visual = Node3D.new()
	stock_visual.position = STACK
	add_child(stock_visual)
	for i in range(48):
		var card = Card.new()
		add_child(card)
		card.position = STACK + Vector3(0, i * CARD_LAYER, 0)
		card.rotation = BACK_FLAT
		card.stow()
		card.arrived.connect(card_arrived)
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
	b.pressed.connect(func(): play_sound("ui-click"); callback.call())
	return b

func make_hud() -> void:
	# Round, scores and turn status inhabit the same upright world plane as decisions.
	left_label = label3("", Vector3(-5.65, 2.0, 0.0), 26, CREAM)
	right_label = label3("", Vector3(5.65, 2.0, 0.0), 26, CREAM)
	bottom_label = label3("", Vector3(0, 1.32, 0.0), 25, GOLD)
	bottom_label.visible = false
	announcement = label3("", Vector3(0, 4.8, 0), 28, CREAM)
	announcement.double_sided = true
	announcement.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
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

func text_size(text: String, size: int) -> Vector2:
	return FONT.get_multiline_string_size(text, HORIZONTAL_ALIGNMENT_CENTER, -1, size) * 0.0085

func panel(title: String, subtitle: String = "", reverse: String = "", _height: float = 0.9, mirrored: bool = false) -> void:
	clear_menu()
	var front_size := text_size(title, 46)
	var reverse_size := text_size(reverse if reverse != "" else title, 46)
	var width := maxf(front_size.x, reverse_size.x) + 0.65
	var height := maxf(front_size.y, reverse_size.y) + 0.24
	plane(Vector2(width, height), Vector3.ZERO, Color("3d301c"), menu, Color("f2bc58"))
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
	var measured := text_size(text, 29) + Vector2(0.6, 0.22)
	width = maxf(width, measured.x)
	var plate := plane(Vector2(width, measured.y), Vector3(x, y, 0.008), Color("49351e"), menu, GOLD)
	var caption := label3(text, Vector3(x, y, 0.013), 29, CREAM, menu)
	menu_buttons.append({"x": x, "y": y, "width": width, "height": measured.y, "action": action, "value": value, "plate": plate, "caption": caption, "fill": Color("49351e"), "ink": CREAM})

func show_title() -> void:
	panel("START")
	menu_buttons.append({"x": 0.0, "y": 0.0, "width": (menu.get_child(0).mesh as QuadMesh).size.x, "height": (menu.get_child(0).mesh as QuadMesh).size.y, "action": "start", "value": 0, "plate": menu.get_child(0), "caption": menu.get_child(1), "fill": menu.get_child(0).material_override.albedo_color, "ink": menu.get_child(1).modulate})
	refresh_hud()

func switch_language() -> void:
	language = 1 - language
	for label in find_children("*", "Label3D", true, false):
		label.font = FONT
	for plaque in label_nodes:
		var text: Label3D = plaque.get_child(1)
		text.text = role_title(plaque.get_meta("roles"))
		var size := text_size(text.text, 18) + Vector2(0.35, 0.16)
		plaque.position.x = (3.6 + size.x / 2) * int(plaque.get_meta("direction"))
		var old_plate: Node = plaque.get_child(0)
		plaque.remove_child(old_plate)
		old_plate.queue_free()
		var new_plate := plane(size, Vector3.ZERO, Color("3d301c"), plaque, GOLD)
		plaque.move_child(new_plate, 0)
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
	bottom_label.visible = false
	left_label.font = FONT
	right_label.font = FONT
	announcement.font = FONT
	var pts: Array = state.points
	left_label.text = "%s    %02d / %02d\n%s  %d %s%s" % [word("round"), state.round, state.get("rounds", 12), word("you"), pts[0], word("points"), "  · " + word("dealer") if state.dealer == 0 else ""]
	right_label.text = "%s  %d %s%s\n%s" % [word("ai"), pts[1], word("points"), "  · " + word("dealer") if state.dealer == 1 else "", word("turn_you") if state.player == 0 else word("turn_ai")]
	if paused:
		right_label.text = right_label.text.get_slice("\n", 0) + "\n" + word("final" if finished else "round_complete")

func send(path: String, data: Dictionary = {}) -> void:
	if busy:
		return
	busy = true
	update_hand_affordance()
	data["revision"] = revision
	if path == "/start":
		data["rounds"] = match_rounds
		selected_hand = -1
		queued_capture = -1
		active_pending = -1
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
	if queued_capture >= 0 and not paused and int(state.phase) == 1 and has_id(state.legal, queued_capture):
		var chosen := queued_capture
		queued_capture = -1
		send("/action", {"action": chosen})
		return
	if not paused and int(state.player) == 0 and int(state.phase) in [1, 2] and equivalent_targets(state.legal):
		send("/action", {"action": int(state.legal[0])})
		return
	show_decision()
	refresh_hud()
	if not state.is_empty():
		await screenshot("round_end" if paused else "koi" if state.phase == 3 else "table")
	if autoplay and finished:
		print("GAME_QA_COMPLETE rounds=%d points=%s bodies=%d" % [state.get("rounds", 12), state.points, bodies.size()])
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
		clear_menu()
		menu.position.y = 3.7
		var question := word("koi_question")
		var question_size := text_size(question, 28) + Vector2(0.6, 0.25)
		plane(question_size, Vector3(0, 0.75 + question_size.y / 2, 0.008), Color("3d301c"), menu, GOLD)
		var question_label := label3(question, Vector3(0, 0.75 + question_size.y / 2, 0.013), 28, CREAM, menu)
		question_label.double_sided = true
		var free_width := right_label.position.x - text_size(right_label.text, right_label.font_size).x / 2 - left_label.position.x - text_size(left_label.text, left_label.font_size).x / 2 - 0.4
		menu.scale = Vector3.ONE * minf(1.0, free_width / maxf(question_size.x, 5.3))
		if decision_announced_revision != revision:
			play_sound("ui-message")
			decision_announced_revision = revision
		world_button(word("yes"), 0.0, "action", 49, -1.4, 2.5)
		world_button(word("no"), 0.0, "action", 48, 1.4, 2.5)
		for child in menu.get_children():
			if child is Label3D:
				child.double_sided = true
		if int(state.player) == 1:
			menu.rotation.y = PI
			menu_buttons.clear()
	else:
		clear_menu()
	update_hand_affordance()

func player_name(player: int) -> String:
	return ("プレイヤー" if player == 0 else "AI") if language == 0 else ("Player" if player == 0 else "AI")

func card_name(id: int) -> String:
	var special := {0: "松に鶴", 4: "梅に鶯", 8: "桜に幕", 12: "藤に不如帰", 16: "菖蒲に八橋", 20: "牡丹に蝶", 24: "萩に猪", 28: "芒に月", 29: "芒に雁", 32: "菊に盃", 36: "紅葉に鹿", 40: "柳に小野道風", 41: "柳に燕", 44: "桐に鳳凰"}
	if language == 0:
		return special.get(id, JP_MONTHS[id / 4] + ("の短冊" if Layout.category(id) == 2 else "のかす"))
	return MONTHS[id / 4] + " · " + ["Chaff", "Animal", "Ribbon", "Light"][Layout.category(id)]

func announce(message: String, seconds: float = 0.9, recipient: int = 0) -> void:
	announcement.rotation.y = PI if recipient == 1 else 0
	play_sound("ui-message")
	announcement.text = message
	var viewport_size := get_viewport().get_visible_rect().size
	var depth := maxf(1.0, -camera.to_local(announcement.global_position).z)
	var available_width := 2.0 * depth * tan(deg_to_rad(camera.fov / 2)) * viewport_size.x / viewport_size.y * 0.85
	announcement.width = ceili(minf(text_size(message, announcement.font_size).x, available_width) / announcement.pixel_size)
	announcement_history.append(message)
	await wait_minimum(seconds, 1.0)
	announcement.text = ""

func outcome_message(player: int, won: bool, stopped: bool) -> String:
	if language == 1:
		return player_name(player) + (" WIN" if won else " LOSE")
	if won:
		return player_name(player) + "さんの勝ちです"
	return (player_name(1 - player) + "さんがこいこいしませんでした。" if stopped else "") + player_name(player) + "さんの負けです。"

func announce_outcome(winner: int, stopped: bool) -> void:
	var faces: Array[Label3D] = []
	play_sound("ui-message")
	for player in range(2):
		var message := outcome_message(player, player == winner, stopped)
		var face := label3(message, announcement.position + Vector3(0, 0, 0.012 if player == 0 else -0.012), announcement.font_size, CREAM, self)
		face.name = "OutcomeFront" if player == 0 else "OutcomeBack"
		face.double_sided = false
		face.rotation.y = PI if player == 1 else 0
		var viewport_size := get_viewport().get_visible_rect().size
		var depth := maxf(1.0, -camera.to_local(face.global_position).z)
		var available_width := 2.0 * depth * tan(deg_to_rad(camera.fov / 2)) * viewport_size.x / viewport_size.y * 0.85
		face.width = ceili(minf(text_size(message, face.font_size).x, available_width) / face.pixel_size)
		faces.append(face)
		announcement_history.append(message)
	await wait_minimum(1.25, 1.25)
	for face in faces:
		face.queue_free()

func update_hand_affordance() -> void:
	if state.is_empty():
		return
	for slot in range(8):
		var id := int(state.hand[slot])
		if id < 0 or id == active_pending or not known.has(id):
			continue
		var card = known[id]
		var raised: bool = not busy and not modal and not paused and int(state.player) == 0 and int(state.phase) == 0 and has_id(state.legal, id) and not matching_field(id).is_empty()
		if selected_hand >= 0:
			raised = id == selected_hand
		var target := hand_position(slot, 0) + Vector3(0, 0.24 if raised else 0, 0)
		if card.position.distance_to(target) > 0.005:
			card.move_to(target, Vector3.ZERO, 0.16 / speed)

func has_id(values: Array, id: int) -> bool:
	return values.any(func(value): return int(value) == id)

func matching_field(id: int) -> Array:
	return state.field.filter(func(c): return int(c) / 4 == id / 4)

func equivalent_targets(ids: Array) -> bool:
	if ids.size() < 2:
		return false
	return ids.all(func(id): return int(id) / 4 == int(ids[0]) / 4 and Layout.category(int(id)) == Layout.category(int(ids[0])))

func cancel_selection() -> void:
	selected_hand = -1
	queued_capture = -1
	update_hand_affordance()

func handle_table_click(screen: Vector2) -> void:
	if click_menu(screen) or busy or not started or paused or int(state.player) != 0:
		return
	var card = hit_card(screen)
	if selected_hand >= 0:
		if card != null and has_id(matching_field(selected_hand), card.identity):
			var chosen := selected_hand
			queued_capture = card.identity if matching_field(chosen).size() == 2 else -1
			selected_hand = -1
			active_pending = chosen
			send("/action", {"action": chosen})
			return
		if card == null or card.identity == selected_hand or not has_id(state.hand, card.identity):
			cancel_selection()
			return
	if card != null and has_id(state.legal, card.identity):
		if int(state.phase) == 0 and matching_field(card.identity).size() >= 2 and not equivalent_targets(matching_field(card.identity)):
			selected_hand = card.identity
			update_hand_affordance()
		else:
			active_pending = card.identity if int(state.phase) == 0 else active_pending
			send("/action", {"action": card.identity})

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
	var middle := ceili(lines.size() / 2.0)
	var parts: Array = ["\n".join(lines.slice(0, middle)), "\n".join(lines.slice(middle))] if columns else [text]
	var font_size := 23 if columns else 25
	var sizes: Array[Vector2] = []
	var width := 0.6
	var height := 0.0
	for part in parts:
		var measured := text_size(part, font_size)
		sizes.append(measured)
		width += measured.x + 0.4
		height = maxf(height, measured.y + 0.5)
	panel(headline, word("final") if finished else "%s %d / 12" % [word("round"), state.round], "LOSE" if headline == "WIN" else "WIN" if headline == "LOSE" else "DRAW")
	menu.get_child(0).material_override = Card.material(Color("b94013"))
	for child in menu.get_children():
		if child is Label3D and child.font_size == 46:
			child.modulate = CREAM
			child.outline_modulate = Color("18797d")
	menu.position.y = 3.8 + maxf(0, height - 1.6) * 0.5
	var top := -maxf(1.05, (menu.get_child(0).mesh as QuadMesh).size.y / 2 + text_size(word("final"), 25).y + 0.35)
	plane(Vector2(width, height), Vector3(0, top - height / 2, 0.002), Color("3d301c"), menu, GOLD)
	var x := -width / 2 + 0.5
	for i in range(parts.size()):
		label3(parts[i], Vector3(x + sizes[i].x / 2, top - height / 2, 0.006), font_size, CREAM, menu)
		x += sizes[i].x + 0.4
	world_button(word("new" if finished else "next"), top - height - text_size(word("new" if finished else "next"), 29).y / 2 - 0.22, "start" if finished else "next")
	# The content determines its plate size; fit the entire result between the
	# two persistent score readouts instead of clipping or shrinking a text box.
	var free_width := right_label.position.x - text_size(right_label.text, right_label.font_size).x / 2 - left_label.position.x - text_size(left_label.text, left_label.font_size).x / 2 - 0.4
	var factor := minf(1.0, free_width / width)
	menu.scale = Vector3.ONE * factor
	menu.position.y = 3.8 + maxf(0, height * factor - 1.6) * 0.5


func maybe_ai() -> void:
	if busy or modal or state.is_empty() or finished:
		return
	if autoplay and paused:
		send("/next")
	elif not paused and int(state.player) == 1:
		await wait_minimum(0.55, 1.2 if int(state.phase) == 3 else 0.8)
		if not busy and not modal:
			send("/ai")
	elif autoplay and not paused:
		var actions: Array = state.legal
		if not actions.is_empty():
			send("/action", {"action": int(actions[0])})

func wait_minimum(seconds: float, minimum: float) -> void:
	# Functional QA may accelerate the clock; separate presentation tests
	# exercise the actual minimum duration without this environment flag.
	if testing and OS.get_environment("KOIKOI_QA_FAST") == "1":
		minimum = 0.0
	await get_tree().create_timer(maxf(minimum, seconds / speed)).timeout

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

func configure_field(ids: Array) -> void:
	for id in field_slots.keys():
		if not has_id(ids, int(id)):
			field_slots.erase(id)
	for raw_id in ids:
		var id := int(raw_id)
		if not field_slots.has(id):
			var slot := 0
			while slot in field_slots.values():
				slot += 1
			field_slots[id] = slot
			field_columns = maxi(field_columns, ceili((slot + 1) / 2.0))
	board_left = Layout.field_left(field_columns * 2)
	var width := 5.3 - board_left
	board.scale.x = width / 10.6
	board.position.x = (5.3 + board_left) / 2
	board_body.position.x = board.position.x
	board_shape.size.x = width

func field_position(card_id: int) -> Vector3:
	if not field_slots.has(card_id):
		configure_field(field_slots.keys() + [card_id])
	return Layout.field(int(field_slots[card_id]), field_columns * 2)

func slide_field(ids: Array, except_id: int = -1) -> void:
	configure_field(ids)
	for raw_id in ids:
		var id := int(raw_id)
		if id == except_id or not known.has(id):
			continue
		var card = known[id]
		var pos := field_position(id)
		if card.position.distance_to(pos) > 0.008:
			fly(card, pos, FLAT, 0.62)

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
			if other == card or card.capture_slide or card.pair_peer != null or other.pair_peer == card:
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
		if ar.capture_slide != br.capture_slide:
			return ar.capture_slide
		if ar.capture_slide and br.capture_slide and ar.flight_to.y != br.flight_to.y:
			return ar.flight_to.y < br.flight_to.y
		if ar == br:
			return a.flight_from.y > b.flight_from.y
		if absf(ar.flight_to.z) > 11.0 and absf(br.flight_to.z) > 11.0 and ar.flight_to.z * br.flight_to.z > 0 and ar.flight_to.y != br.flight_to.y:
			return ar.flight_to.y < br.flight_to.y
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
			if card.role_route and card.role_clearance_weight((t * card.duration - card.start_delay) / (card.duration - card.start_delay)) < 0.001:
				continue
			var pose: Transform3D = card.pose_at(t * card.duration)
			var extent := half_bounds(pose.basis)
			var floor_y := -100.0
			var ceiling_y := 100.0
			if pose.origin.x > board_left - extent.x and pose.origin.x < 5.3 + extent.x and absf(pose.origin.z) < 2.85 + extent.z:
				floor_y = 0.006
			# The solid stock participates as one box, never 48 exposed slabs.
			if not card.bottom_return and absf(pose.origin.x - STACK.x) < extent.x + Card.SIZE.x / 2 and absf(pose.origin.z - STACK.z) < extent.z + Card.SIZE.y / 2:
				floor_y = maxf(floor_y, stack.size() * CARD_LAYER + 0.006)
			for other in bodies:
				if other == card or other in lower_priority or not other.visible or card.pair_peer == other or other.pair_peer == card:
					continue
				var other_pose: Transform3D = other.pose_at(other.elapsed + t * card.duration) if other.flight else other.transform
				var other_extent := half_bounds(other_pose.basis)
				var role_t: float = clampf((t * card.duration - card.start_delay) / (card.duration - card.start_delay), 0, 1)
				var in_sideways_leg: bool = card.role_route and role_t * (card.role_lengths.x + card.role_lengths.y + card.role_lengths.z) < card.role_lengths.x
				if card.role_route and not Card.overlaps(pose, other_pose, 0.001 if in_sideways_leg else 0.03):
					continue
				var diff := (pose.origin - other_pose.origin).abs()
				if diff.x < extent.x + other_extent.x + 0.03 and diff.z < extent.z + other_extent.z + 0.03:
					if card.capture_slide:
						if other_pose.origin.y - other_extent.y > 0.25:
							continue
						var other_y: float = other.flight_to.y if other.flight else other_pose.origin.y
						if other_y > card.flight_to.y + 0.005:
							ceiling_y = minf(ceiling_y, other_pose.origin.y - other_extent.y - 0.001)
							continue
					# Passing beneath a separated card is valid; lift only for a
					# vertical conflict, including a sampling/clearance margin.
					if card.capture_slide or (minf(card.flight_from.z, card.flight_to.z) < other_pose.origin.z and maxf(card.flight_from.z, card.flight_to.z) > other_pose.origin.z and other_extent.y > 0.5 and (is_equal_approx(other_pose.origin.z, 3.9) or is_equal_approx(other_pose.origin.z, -4.25)) and (not other.flight or Vector2(other.flight_from.x, other.flight_from.z).distance_to(Vector2(other.flight_to.x, other.flight_to.z)) < 0.001)) or (pose.origin.y + extent.y >= other_pose.origin.y - other_extent.y - 0.08 and pose.origin.y - extent.y <= other_pose.origin.y + other_extent.y + 0.08):
						floor_y = maxf(floor_y, other_pose.origin.y + other_extent.y + (0.001 if card.capture_slide or (returning_all and t > 0.9) else 0.025))
			var deficit := floor_y + extent.y - pose.origin.y
			if deficit > 0:
				card.clearance[i] += deficit + 0.003
				changed = true
			if card.capture_slide:
				var adjusted: Transform3D = card.pose_at(t * card.duration)
				var excess := adjusted.origin.y + extent.y - ceiling_y
				if excess > 0:
					card.clearance[i] -= excess + 0.001
					changed = true
		if card.capture_slide:
			if not changed:
				break
			continue
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
		if card.pair_peer != null:
			# Both cards share clearance while they are still touching. Update
			# the leader now so the follower's separation constraint cannot
			# repeatedly cancel the lift and make the solver accumulate height.
			for sample in range(1, 96):
				var shared: float = maxf(card.clearance[sample], card.pair_peer.clearance[sample])
				card.clearance[sample] = shared
				card.pair_peer.clearance[sample] = shared
		if not changed:
			break

func finish_flights() -> void:
	plan_flights()
	while bodies.any(func(c): return c.flight):
		await get_tree().physics_frame

func reset_deck(separate_cards: bool = false) -> void:
	clear_yaku_labels()
	known.clear()
	opponent_cards.clear()
	field_slots.clear()
	field_columns = 4
	capture_slots = [{}, {}]
	role_order = [[], []]
	role_slots = [{}, {}]
	pending_flights.clear()
	selected_hand = -1
	queued_capture = -1
	active_pending = -1
	var returning: Array = bodies.filter(func(c): return c.visible)
	var groups: Dictionary = {}
	returning.sort_custom(func(a, b): return a.position.y > b.position.y)
	if separate_cards:
		returning.sort_custom(func(a, b): return a.position.x > b.position.x)
	for card in returning:
		var z: float = card.position.z
		var key: String = "role_%s_%s" % [signf(z), roundi(card.position.y / 1.52)] if absf(z) > 11 else "capture_%s_%s" % [signf(z), roundi(absf(z))] if absf(z) > 2.85 and absf(card.basis.z.y) > 0.5 else "hand_%s" % signf(z) if absf(z) > 3.5 else "field"
		if Vector2(card.position.x - STACK.x, card.position.z - STACK.z).length() < 0.05:
			key = "stock"
		if separate_cards:
			key = "single_%s" % card.get_instance_id()
		if not groups.has(key):
			groups[key] = []
		groups[key].append(card)
	returning_all = true
	var packets: Array = []
	var group_id := 0
	for group in groups.values():
		# Pick an existing outer corner nearest the stock. Gathering uses the
		# same plane and adds no stack height; merged cards become one visual.
		group.sort_custom(func(a, b): return a.position.x > b.position.x if not is_equal_approx(a.position.x, b.position.x) else a.position.distance_to(STACK) < b.position.distance_to(STACK))
		var anchor = group[0]
		var target: Vector3 = anchor.position
		var angles: Vector3 = anchor.rotation
		for card in group:
			card.pair_peer = null
			card.move_to(target, angles, 0.38 / speed)
			card.gather_group = group_id
		packets.append({"anchor": anchor, "members": group, "bottom": absf(anchor.basis.z.y) > 0.5 and Vector2(anchor.position.x - STACK.x, anchor.position.z - STACK.z).length() > 0.05})
		group_id += 1
	await finish_flights()
	for packet in packets:
		for card in packet.members:
			if card != packet.anchor:
				card.stow()
		packet.anchor.gather_group = -1
	packets.sort_custom(func(a, b): return a.bottom and not b.bottom)
	for packet in packets:
		var card = packet.anchor
		var target_angles: Vector3 = card.rotation
		if absf(card.basis.z.y) < 0.5:
			target_angles = BACK_FLAT
		# Flat packets slide under the deck. Lift the deck, never the incoming
		# cards; its top stays continuous when the packet merges into the prism.
		if packet.bottom:
			var gap: float = maxf(packet.members.size() * CARD_LAYER, card.position.y + 0.025)
			create_tween().tween_property(stock_visual, "position:y", gap, 0.12 / speed)
		var landing_y: float = 0.012 if packet.bottom else stack.size() * CARD_LAYER + 0.012
		fly(card, STACK + Vector3(0, landing_y, 0), target_angles, 0.6)
		card.returning = true
		card.bottom_return = packet.bottom
		play_sound("fuda-turn", card.identity)
		await finish_flights()
		for member in packet.members:
			member.stow()
			member.identity = -1
			member.set_face(null)
			if packet.bottom:
				stack.push_front(member)
			else:
				stack.append(member)
		if packet.bottom:
			stock_visual.position.y = maxf(0, stock_visual.position.y - packet.members.size() * CARD_LAYER)
		update_stock()
		if stock_visual.position.y > 0.001:
			await create_tween().tween_property(stock_visual, "position:y", 0.0, 0.12 / speed).finished
	stock_visual.position.y = 0.0
	returning_all = false
	configure_field([])

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
			card.arrival_sound = "tefuda-place"
			await finish_flights()
		await wait_for(0.3)
		# Dealer reveal is the one explicitly requested stationary turn.
		for card in pair:
			card.move_to(card.position, card.rotation + Vector3(0, PI, 0), 1.1 / speed)
		await finish_flights()
		clear_menu()
		if int(reveal.winner) < 0:
			await announce(word("tie"))
		else:
			var names := card_name(int(reveal.cards[0])) + "、" + card_name(int(reveal.cards[1]))
			await announce(names + "。親が" + player_name(int(reveal.winner)) + "さんに決定しました。" if language == 0 else names + ". " + player_name(int(reveal.winner)) + " is the dealer.", 1.5)
		await reset_deck(true) # Dealer cards return singly, without merging.
	clear_menu()
	await deal_cards(event.view, false)

func deal_cards(view: Dictionary, returning: bool = true) -> void:
	if returning:
		await reset_deck()
	clear_menu()
	state = view
	configure_field(view.field)
	refresh_hud()
	for batch in range(4):
		for group in [2, int(view.dealer), 1 - int(view.dealer)]:
			# Preserve the field/dealer/opponent two-card batches, dealt singly.
			for j in range(2):
				var slot := batch * 2 + j
				var id := int(view.field[slot]) if group == 2 else int(view.hand[slot]) if group == 0 else -1
				var card = take_card(id)
				var destination: Vector3
				if group == 2:
					card.rotation = FLAT
					destination = field_position(id)
				else:
					card.slot = slot
					card.seat = group
					if group == 1:
						opponent_cards[slot] = card
					destination = hand_position(slot, group)
				fly(card, destination, FLAT if group == 2 else Vector3(0, PI * group, 0), 0.42)
				card.arrival_sound = "bafuda-place" if group == 2 else "tefuda-place"
				await finish_flights()

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
		active_pending = id
		pending_origin = card.position
		if int(event.player) == 1:
			card.move_to(card.position, Vector3.ZERO, 0.42 / speed)
			await finish_flights()
		pending_angles = card.rotation
	elif kind == "draw":
		var id := int(event.card)
		var card = take_card(id)
		play_sound("fuda-turn", id)
		card.rotation = FLAT # The specified exception: instantaneous stock-top flip.
		active_pending = id
		pending_origin = card.position
		pending_angles = card.rotation
	elif kind in ["capture", "place"]:
		var card = known[int(event.card)]
		if kind == "place":
			slide_field(view.field, int(event.card))
		var target: Vector3 = field_position(int(event.card)) if kind == "place" else known[int(event.targets[0])].position + Vector3(0.13, 0.032, 0.13)
		fly(card, target + Vector3(0, 0.48 if event.new_yaku else 0, 0), FLAT, 0.30)
		card.arrival_sound = "" if event.new_yaku else "bafuda-place"
		await finish_flights()
		if event.new_yaku:
			await impact_capture(card, event.targets)
		active_pending = -1
		if kind == "capture":
			slide_field(view.field)
			await arrange_captures(view)
			if event.new_yaku:
				var changed_roles: Array = view.yaku[int(event.player)].filter(func(row): return not row.cards.is_empty() and not state.yaku[int(event.player)].any(func(old): return old.key == row.key and old.points >= row.points))
				for row in changed_roles:
					await announce(word(row.key) + "です。" if language == 0 else word(row.key) + ".", 1.0, int(event.player))


	elif kind == "koi":
		if int(event.player) == 1:
			play_sound("ui-click")
		var name := player_name(int(event.player))
		await announce(name + "さん、こいこいです。" if language == 0 else name + " calls Koikoi", 0.9, 1 - int(event.player))
	elif kind == "round_end":
		var winner := int(event.winner)
		play_sound("win" if winner == 0 else "lose")
		await announce_outcome(winner, event.reason == "stop")
		if event.total_yaku:
			announcement.rotation.y = 0
			announcement.text = "得点が移動します。" if language == 0 else "The points are being transferred."
			announcement_history.append(announcement.text)
			await sweep_yaku(view, winner)
			announcement.text = ""
			await screenshot("total_yaku")
	state = view
	refresh_hud()

func restore_view(view: Dictionary) -> void:
	# Recover public objects through visible, collision-aware flights.
	await reset_deck()
	configure_field(view.field)
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
		var pending = take_card(int(view.pending))
		active_pending = int(view.pending)
		if int(view.phase) == 1:
			pending.slot = int(view.get("pending_slot", 0))
			fly(pending, hand_position(pending.slot, int(view.player)), Vector3(0, PI if int(view.player) == 0 else 0, 0), 0.3)
			await finish_flights()
		else:
			pending.rotation = FLAT
		pending_origin = pending.position
		pending_angles = pending.rotation
	await arrange_captures(view, true)
	if paused and ending.get("total_yaku", false):
		await sweep_yaku(view, int(ending.winner))

func clear_yaku_labels() -> void:
	for node in label_nodes:
		node.queue_free()
	label_nodes.clear()

func role_key(key: String) -> String:
	return "lights" if key in ["three", "four", "rain_four", "five"] else key

func presentation_roles(view: Dictionary, player: int) -> Array:
	# Connected components: a physical card belongs to one row, including
	# transitive overlaps such as lights + flower viewing + moon viewing.
	var rows: Array = view.yaku[player].filter(func(row): return not row.cards.is_empty())
	for row in rows:
		var key := role_key(row.key)
		if key not in role_order[player]:
			role_order[player].append(key)
			role_slots[player][key] = {}
	rows.sort_custom(func(a, b): return role_order[player].find(role_key(a.key)) < role_order[player].find(role_key(b.key)))
	var groups: Array = []
	while not rows.is_empty():
		var first: Dictionary = rows.pop_front()
		var group := {"key": first.key, "points": first.points, "cards": first.cards.duplicate(), "roles": [first]}
		var expanded := true
		while expanded:
			expanded = false
			for i in range(rows.size() - 1, -1, -1):
				var row: Dictionary = rows[i]
				if row.cards.any(func(id): return id in group.cards):
					group.roles.append(row)
					group.points += row.points
					for id in row.cards:
						if id not in group.cards:
							group.cards.append(id)
					rows.remove_at(i)
					expanded = true
		group.roles.sort_custom(func(a, b): return role_order[player].find(role_key(a.key)) < role_order[player].find(role_key(b.key)))
		groups.append(group)
	return groups

func role_title(roles: Array) -> String:
	var titles: PackedStringArray = []
	for row in roles:
		titles.append(word(row.key) + "  " + str(row.points))
	return " ／ ".join(titles)

func arrange_captures(view: Dictionary, serial: bool = false) -> void:
	clear_yaku_labels()
	var changed := false
	for player in range(2):
		var allocated: Array = []
		var row_index := 0
		var rows := presentation_roles(view, player)
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
			var z := 12.2 if player == 0 else -12.2
			var y := 3.55 - row_index * 1.52
			var plaque := Node3D.new()
			plaque.set_meta("roles", row.roles)
			plaque.set_meta("yaku_key", row.key)
			plaque.set_meta("points", row.points)
			plaque.set_meta("direction", 1 if player == 0 else -1)
			var plaque_text := role_title(row.roles)
			var plaque_size := text_size(plaque_text, 18) + Vector2(0.35, 0.16)
			plaque.position = Vector3((3.6 + plaque_size.x / 2) * (1 if player == 0 else -1), y, z)
			plaque.rotation.y = PI if player == 0 else 0.0
			add_child(plaque)
			plane(plaque_size, Vector3.ZERO, Color("3d301c"), plaque, GOLD)
			label3(plaque_text, Vector3(0, 0, 0.006), 18, GOLD, plaque)
			label_nodes.append(plaque)
			var arrivals: Array = []
			for i in range(cards_here.size()):
				var card = known[cards_here[i]] if known.has(cards_here[i]) else take_card(cards_here[i])
				var slot: int = slots[cards_here[i]]
				var spacing := 0.88
				var x := (3.0 - slot * spacing) * (1 if player == 0 else -1)
				var pos := Vector3(x, y, z)
				# Real cards occupy just one role row; overlapping roles share membership.
				if card.position.distance_to(pos) > 0.015:
					if arrivals.is_empty():
						play_sound("fuda-take")
					fly(card, pos, Vector3(0, PI if player == 0 else 0.0, 0), 0.85 + slot * 0.055)
					card.arrival_sound = "yaku"
					arrivals.append(card)
					if absf(card.flight_from.z) < 10.0:
						card.route_to_role(player, absf(board_left) + 1.2 if player == 0 else 6.5)
					changed = true
					if serial:
						await finish_flights()
			if not serial:
				stagger_roles(arrivals)
			row_index += 1
		for id in allocated:
			capture_slots[player].erase(id)
		var incoming: Array = view.captured[player].duplicate()
		incoming.sort_custom(func(a, b): return known[int(a)].position.y < known[int(b)].position.y if known.has(int(a)) and known.has(int(b)) else int(a) < int(b))
		for raw_id in incoming:
			var id := int(raw_id)
			if id in allocated:
				continue
			if not capture_slots[player].has(id):
				var used: Array = capture_slots[player].keys().filter(func(c): return Layout.category(int(c)) == Layout.category(id)).map(func(c): return capture_slots[player][c])
				var free := 0
				while free in used:
					free += 1
				capture_slots[player][id] = free
		var counts := [0, 0, 0, 0]
		for id in capture_slots[player]:
			var kind := Layout.category(int(id))
			counts[kind] = maxi(counts[kind], int(capture_slots[player][id]) + 1)
		for raw_id in view.captured[player]:
			var id := int(raw_id)
			if id in allocated:
				continue
			var pos := Layout.capture(id, int(capture_slots[player][id]), player, counts)
			var card = known[id] if known.has(id) else take_card(id)
			if card.position.distance_to(pos) > 0.015:
				fly(card, pos, FLAT if player == 0 else Vector3(-PI / 2, 0, PI), 0.55)
				card.arrival_sound = "fuda-turn"
				card.capture_slide = absf(card.flight_from.y) < 0.1
				changed = true
				if serial:
					await finish_flights()
	if changed:
		await finish_flights()

func stagger_roles(cards: Array) -> void:
	var common_edge := 0.0
	for card in cards:
		if card.role_route:
			common_edge = maxf(common_edge, absf(card.role_edge))
	for card in cards:
		if card.role_route:
			card.route_to_role(0 if card.role_side < 0 else 1, common_edge)
	var base := 0.0
	var times: Array[float] = []
	for card in cards:
		var length: float = card.role_lengths.x + card.role_lengths.y + card.role_lengths.z if card.role_route else card.flight_from.distance_to(card.flight_to)
		var seconds := maxf(0.45, length / 14.0)
		times.append(seconds)
		base = maxf(base, seconds)
	for i in range(cards.size()):
		cards[i].duration = (base + i * 0.1) / speed
		cards[i].start_delay = (base + i * 0.1 - times[i]) / speed

func sweep_yaku(view: Dictionary, winner: int) -> void:
	var loser := 1 - winner
	var direction := 1 if winner == 0 else -1
	# The winner's cards and plaques keep their exact settled transforms.
	for i in range(label_nodes.size() - 1, -1, -1):
		if int(label_nodes[i].get_meta("direction", direction)) != direction:
			label_nodes[i].queue_free()
			label_nodes.remove_at(i)
	var allocated: Array = []
	var row_index: int = presentation_roles(view, winner).size()
	var rows := presentation_roles(view, loser)
	for row in rows:
		var ids: Array = []
		for raw_id in row.cards:
			var id := int(raw_id)
			if id not in allocated:
				allocated.append(id)
				ids.append(id)
		var y := 3.55 - row_index * 1.52
		var title := role_title(row.roles)
		var size := text_size(title, 18) + Vector2(0.35, 0.16)
		var plaque := Node3D.new()
		plaque.set_meta("roles", row.roles)
		plaque.set_meta("yaku_key", row.key)
		plaque.set_meta("points", row.points)
		plaque.set_meta("direction", direction)
		plaque.position = Vector3((3.6 + size.x / 2) * direction, y, 12.2 * direction)
		plaque.rotation.y = PI if winner == 0 else 0
		add_child(plaque)
		plane(size, Vector3.ZERO, Color("3d301c"), plaque, GOLD)
		label3(title, Vector3(0, 0, 0.006), 18, GOLD, plaque)
		label_nodes.append(plaque)
		var slots: Dictionary = role_slots[loser].get(role_key(row.key), {})
		play_sound("fuda-take") # One cue per transferred role, never per card.
		for i in range(ids.size()):
			var slot := int(slots.get(ids[i], i))
			var target := Vector3((3.0 - slot * 0.88) * direction, y, 12.2 * direction)
			fly(known[ids[i]], target, Vector3(0, PI if winner == 0 else 0, 0), 1.2, 0.0, true)
		row_index += 1
	await finish_flights()

func impact_capture(card, targets: Array) -> void:
	var previous_ticks := Engine.physics_ticks_per_second
	Engine.physics_ticks_per_second = 480
	# Release the contacted cards before striking. Every upward motion comes
	# from rigid-body contact restitution, never a scripted rebound impulse.
	var struck: Array = [card]
	for id in targets.slice(0, 1):
		if known.has(int(id)):
			struck.append(known[int(id)])
	for body in struck:
		body.flight = false
		body.freeze = false
		body.sleeping = false
		body.physics_material_override.bounce = 0.95
		body.linear_damp = 0.2
	card.strike()
	last_burst_peak = 0
	var elapsed := 0.0
	var contact_joint: PinJoint3D
	var settled_frames := 0
	var first_rebound := false
	last_impact_settled = false
	while elapsed < 5.0:
		await get_tree().physics_frame
		elapsed += 1.0 / Engine.physics_ticks_per_second
		if contact_joint == null and struck.size() == 2 and card.position.y - struck[1].position.y < 0.04 and card.linear_velocity.y < 0:
			# A short-lived contact bond lets the two thin cards share the
			# tabletop's restitution, rather than pinning the lower rigid proxy.
			play_sound("bafuda-place-yaku", card.identity)
			contact_joint = PinJoint3D.new()
			contact_joint.position = struck[1].position
			add_child(contact_joint)
			contact_joint.node_a = card.get_path()
			contact_joint.node_b = struck[1].get_path()
		last_burst_peak = maxf(last_burst_peak, struck.map(func(c): return c.position.y).min())
		if last_burst_peak > 0.1 and struck.all(func(c): return c.linear_velocity.y <= 0):
			first_rebound = true
		if first_rebound:
			for body in struck:
				body.physics_material_override.bounce = 0.12
		if elapsed > 3.0:
			for body in struck:
				body.physics_material_override.bounce = 0.0
				body.linear_damp = 4.0
		var resting: bool = struck.all(func(c): return c.position.y < 0.075 and absf(c.linear_velocity.y) < 0.15)
		settled_frames = settled_frames + 1 if resting else 0
		if settled_frames >= 12:
			last_impact_settled = true
			break
	Engine.physics_ticks_per_second = previous_ticks
	if contact_joint != null:
		contact_joint.queue_free()
	for body in struck:
		body.freeze = true
		body.linear_velocity = Vector3.ZERO
		body.physics_material_override.bounce = 0.38
		body.linear_damp = 1.2

func card_arrived(card) -> void:
	if not card.arrival_sound.is_empty():
		play_sound(card.arrival_sound, card.identity)
		card.arrival_sound = ""

func play_sound(key: String, card_id: int = -1) -> void:
	if testing:
		sound_log.append({"sound": key, "card": card_id, "time": Time.get_ticks_msec()})
	if not sound or autoplay or testing:
		return
	var voice := AudioStreamPlayer.new()
	voice.stream = SOUNDS[key]
	voice.volume_db = -12.0 + float(SOUND_GAIN_DB.get(key, 0.0))
	voice.finished.connect(voice.queue_free)
	add_child(voice)
	voice.play()

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and not modal:
		hover_menu(event.position)
	if event is InputEventKey and event.pressed:
		if event.keycode == KEY_F11:
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED if DisplayServer.window_get_mode() == DisplayServer.WINDOW_MODE_FULLSCREEN else DisplayServer.WINDOW_MODE_FULLSCREEN)
		elif event.keycode == KEY_ESCAPE:
			if selected_hand >= 0:
				cancel_selection()
			elif modal:
				close_overlay()
			else:
				open_overlay("settings")
	if modal:
		overlay_world.route(event, camera)
		return
	if event is InputEventMouseButton:
		if event.pressed and event.button_index in [MOUSE_BUTTON_WHEEL_UP, MOUSE_BUTTON_WHEEL_DOWN]:
			camera_radius = clampf(camera_radius * (0.91 if event.button_index == MOUSE_BUTTON_WHEEL_UP else 1.1), 9.5, 30.0)
			update_camera()
		elif event.button_index in [MOUSE_BUTTON_LEFT, MOUSE_BUTTON_RIGHT]:
			if event.pressed:
				camera_drag = true
				drag_start = event.position
				drag_distance = 0
			else:
				camera_drag = false
				if event.button_index == MOUSE_BUTTON_LEFT and drag_distance < 6:
					handle_table_click(event.position)
	elif event is InputEventMouseMotion and camera_drag:
		drag_distance += event.relative.length()
		if drag_distance >= 6:
			orbit_pitch = clampf(orbit_pitch + event.relative.y * 0.003, deg_to_rad(20), deg_to_rad(56))
			update_camera()

func hit_card(screen: Vector2):
	var ray := PhysicsRayQueryParameters3D.create(camera.project_ray_origin(screen), camera.project_ray_origin(screen) + camera.project_ray_normal(screen) * 80.0, 2)
	var hit := get_world_3d().direct_space_state.intersect_ray(ray)
	return hit.get("collider")

func hover_menu(screen: Vector2) -> void:
	if not is_instance_valid(menu):
		return
	var face := Plane(menu.global_basis.z.normalized(), menu.global_position)
	var point = face.intersects_ray(camera.project_ray_origin(screen), camera.project_ray_normal(screen))
	for item in menu_buttons:
		var hovered := false
		if point != null:
			var local: Vector3 = menu.to_local(point)
			hovered = absf(local.x - item.x) <= item.width / 2 and absf(local.y - item.y) <= item.height / 2
		item.plate.material_override.albedo_color = item.fill.lerp(GOLD, 0.28) if hovered else item.fill
		item.caption.modulate = GOLD if hovered else item.ink

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
			play_sound("ui-click")
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
	update_hand_affordance()
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
	scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
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
	if kind == "restart":
		overlay_box.add_child(ui_label(word("restart_prompt"), 17))
		var restart := ui_button(word("restart"), restart_match)
		restart.disabled = busy
		overlay_box.add_child(restart)
	elif kind == "help":
		var help := ui_label(word("help_body"), 16)
		help.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		overlay_box.add_child(help)
	elif kind == "credits":
		var credits := ui_label("HANAFUDA\nLouie Mantia · すけじょ (traditional colors)\nCC BY-SA 4.0\n\ncommons.wikimedia.org\nFrames recolored to #b44519; illustration ink retained.\n\nAntique Six Extra Bold · Jordan Davies\nNoto Serif JP Black · The Noto Project\nSIL Open Font License 1.1\n\nCard audio · user-supplied clips\nUI audio · original synthesized cues\n\nGodot Engine 4.7.2 · MIT\ngodotengine.org\n\nRules & AI integration\nKoiKoi-AI · Sanghai Guan · MIT\n\nFull notices: game/assets and THIRD_PARTY_NOTICES.md", 17)
		overlay_box.add_child(credits)
	else:
		overlay_box.add_child(ui_button("日本語 / ENGLISH", switch_language))
		overlay_box.add_child(ui_label(word("opponent"), 17, GOLD))
		var options := OptionButton.new()
		options.add_theme_stylebox_override("hover", style(Color("513c37"), GOLD))
		options.add_theme_stylebox_override("pressed", style(Color("6b433d"), GOLD))
		options.add_theme_font_override("font", FONT)
		options.add_theme_font_size_override("font_size", 17)
		var names := ["trained", "rl-wp", "rl-point", "greedy"]
		for name in names:
			options.add_item(word(name) if name in ["trained", "greedy"] else name.to_upper())
		options.select(names.find(opponent))
		options.item_selected.connect(func(i): play_sound("ui-click"); opponent = names[i]; save_settings())
		overlay_box.add_child(options)
		overlay_box.add_child(ui_label(word("match_rounds"), 17, GOLD))
		var rounds_option := OptionButton.new()
		rounds_option.name = "MatchRounds"
		rounds_option.add_theme_font_override("font", FONT)
		rounds_option.add_theme_font_size_override("font_size", 17)
		var lengths := [1, 3, 6, 12]
		for count in lengths:
			rounds_option.add_item(str(count))
		rounds_option.select(lengths.find(match_rounds))
		rounds_option.item_selected.connect(func(i): play_sound("ui-click"); match_rounds = lengths[i]; save_settings())
		overlay_box.add_child(rounds_option)
		overlay_box.add_child(ui_label(word("rules"), 17, GOLD))
		for rule in [{"key": "total", "bit": 16}, {"key": "double", "bit": 8}]:
			var toggle := CheckButton.new()
			toggle.add_theme_stylebox_override("hover", style(Color("513c37"), GOLD))
			toggle.add_theme_stylebox_override("hover_pressed", style(Color("6b433d"), GOLD))
			toggle.text = word(rule.key)
			toggle.add_theme_font_override("font", FONT)
			toggle.add_theme_font_size_override("font_size", 17)
			toggle.button_pressed = (flags & int(rule.bit)) != 0
			toggle.toggled.connect(func(on): play_sound("ui-click"); flags = flags | int(rule.bit) if on else flags & ~int(rule.bit); save_settings())
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
		sound_toggle.add_theme_stylebox_override("hover", style(Color("513c37"), GOLD))
		sound_toggle.add_theme_stylebox_override("hover_pressed", style(Color("6b433d"), GOLD))
		sound_toggle.text = word("sound")
		sound_toggle.add_theme_font_override("font", FONT)
		sound_toggle.button_pressed = sound
		sound_toggle.toggled.connect(func(v): sound = v; play_sound("ui-click"); save_settings())
		overlay_box.add_child(sound_toggle)
		var restart := ui_button(word("restart"), func(): open_overlay("restart"))
		restart.name = "RestartMatch"
		restart.disabled = busy or not started
		overlay_box.add_child(restart)
		overlay_box.add_child(ui_button(word("help"), func(): open_overlay("help")))
		overlay_box.add_child(ui_button(word("credits"), func(): open_overlay("credits")))
	await get_tree().process_frame
	if is_instance_valid(overlay_world) and is_instance_valid(overlay):
		var minimum := overlay.get_combined_minimum_size()
		overlay.size = Vector2(maxf(820, minimum.x), maxf(780, minimum.y))
		overlay_world.fit_content(Vector2i(overlay.size.ceil()), camera)

func restart_match() -> void:
	if busy:
		return
	# Do not resume the old AI turn through close_overlay before /start.
	if is_instance_valid(overlay_world):
		overlay_world.queue_free()
	modal = false
	send("/start", {"opponent": opponent, "flags": flags})

func close_overlay() -> void:
	if is_instance_valid(overlay_world):
		overlay_world.queue_free()
	modal = false
	update_hand_affordance()
	await maybe_ai()

func load_settings() -> void:
	var config := ConfigFile.new()
	if config.load("user://settings.cfg") == OK:
		language = clampi(int(config.get_value("game", "language", 0)), 0, 1)
		speed = clampf(float(config.get_value("game", "speed", 1.0)), 0.6, 2.0)
		sound = bool(config.get_value("game", "sound", true))
		opponent = str(config.get_value("game", "opponent", "trained"))
		flags = int(config.get_value("game", "flags", 0)) & 24
		var saved_rounds := int(config.get_value("game", "rounds", 12))
		match_rounds = saved_rounds if saved_rounds in [1, 3, 6, 12] else 12

func save_settings() -> void:
	if testing:
		return
	var config := ConfigFile.new()
	for pair in [["language", language], ["speed", speed], ["sound", sound], ["opponent", opponent], ["flags", flags], ["rounds", match_rounds]]:
		config.set_value("game", pair[0], pair[1])
	config.save("user://settings.cfg")

func screenshot(name: String) -> void:
	if not qa or qa_frames.has(name) or DisplayServer.get_name() == "headless":
		return
	qa_frames[name] = true
	await get_tree().process_frame
	RenderingServer.force_draw(false)
	var path := ProjectSettings.globalize_path("res://../output/game-qa")
	DirAccess.make_dir_recursive_absolute(path)
	get_viewport().get_texture().get_image().save_png(path.path_join(name + ".png"))

func _process(delta: float) -> void:
	if autoplay:
		elapsed_match += delta
		if elapsed_match > 300:
			push_error("Autoplay watchdog expired")
			get_tree().quit(2)
