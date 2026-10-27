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
const STACK = Vector3(5.1, 0.10, 0.05)
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
var left_label: Label
var right_label: Label
var bottom_label: Label
var overlay: PanelContainer
var overlay_box: VBoxContainer
var inspector: PanelContainer
var inspect_image: TextureRect
var inspect_text: Label
var modal := false
var last_hover := -1
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
			textures.append(load("res://assets/cards/Hanafuda_%s_%s.svg" % [MONTHS[month], kind]))

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
	if parent == null:
		add_child(label)
	else:
		parent.add_child(label)
	return label

func make_world() -> void:
	var environment := WorldEnvironment.new()
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color("b78b39")
	env.tonemap_mode = Environment.TONE_MAPPER_LINEAR
	environment.environment = env
	add_child(environment)
	camera = Camera3D.new()
	camera.position = Vector3(0, 14.6, 15.0)
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 11.7
	add_child(camera)
	camera.look_at(Vector3(0, 0.4, 0.1))
	camera.current = true
	mesh_box(Vector3(12.6, 0.15, 5.65), Vector3(0, -0.10, 0), Color("291324"), self)
	mesh_box(Vector3(12.5, 0.06, 5.55), Vector3(0, 0, 0), Color("4f063d"), self)
	# A real collision surface, confined to the floating board (no ground plane).
	var table_body := StaticBody3D.new()
	var shape := CollisionShape3D.new()
	var box := BoxShape3D.new()
	box.size = Vector3(12.5, 0.12, 5.55)
	shape.shape = box
	table_body.position.y = -0.03
	table_body.collision_layer = 1
	table_body.add_child(shape)
	add_child(table_body)
	for x in [-6.15, 6.15]:
		mesh_box(Vector3(0.015, 0.006, 5.3), Vector3(x, 0.035, 0), GOLD, self)
	for z in [-2.65, 2.65]:
		mesh_box(Vector3(12.3, 0.006, 0.015), Vector3(0, 0.035, z), GOLD, self)
	var mark := label3("KOI  /  KOI", Vector3(-5.0, 0.04, 0), 29, Color("b37399"))
	mark.rotation.x = -PI / 2
	for i in range(48):
		var card = Card.new()
		add_child(card)
		card.position = STACK + Vector3(0, i * 0.047, 0)
		card.rotation = BACK_FLAT
		bodies.append(card)
		stack.append(card)

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
	hud = CanvasLayer.new()
	add_child(hud)
	var root := Control.new()
	root.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	root.mouse_filter = Control.MOUSE_FILTER_IGNORE
	hud.add_child(root)
	left_label = ui_label("", 17, INK)
	left_label.position = Vector2(30, 28)
	root.add_child(left_label)
	right_label = ui_label("", 17, INK)
	right_label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
	right_label.set_anchors_and_offsets_preset(Control.PRESET_TOP_RIGHT)
	right_label.position = Vector2(-350, 28)
	right_label.size.x = 320
	root.add_child(right_label)
	bottom_label = ui_label("", 15, INK)
	bottom_label.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_LEFT)
	bottom_label.position = Vector2(30, -50)
	root.add_child(bottom_label)
	var rail := HBoxContainer.new()
	rail.set_anchors_and_offsets_preset(Control.PRESET_BOTTOM_RIGHT)
	rail.position = Vector2(-270, -62)
	rail.add_theme_constant_override("separation", 8)
	root.add_child(rail)
	rail.add_child(ui_button("?", func(): open_overlay("help")))
	rail.add_child(ui_button("JA / EN", switch_language))
	rail.add_child(ui_button("☰", func(): open_overlay("settings")))
	inspector = PanelContainer.new()
	inspector.position = Vector2(26, 150)
	inspector.add_theme_stylebox_override("panel", style(INK, GOLD))
	var box := VBoxContainer.new()
	inspector.add_child(box)
	inspect_image = TextureRect.new()
	inspect_image.custom_minimum_size = Vector2(148, 243)
	inspect_image.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
	inspect_image.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
	box.add_child(inspect_image)
	inspect_text = ui_label("", 14)
	box.add_child(inspect_text)
	root.add_child(inspector)
	inspector.hide()
	refresh_hud()

func clear_menu() -> void:
	if is_instance_valid(menu):
		menu.queue_free()
	menu_buttons.clear()
	menu = Node3D.new()
	menu.position = Vector3(0, 1.9, 0.5)
	add_child(menu)

func panel(title: String, subtitle: String = "", reverse: String = "", height: float = 3.3, mirrored: bool = false) -> void:
	clear_menu()
	menu.position.y = maxf(1.9, height / 2.0 + 0.35)
	mesh_box(Vector3(7.5, height, 0.08), Vector3.ZERO, GOLD, menu)
	mesh_box(Vector3(7.46, height - 0.04, 0.09), Vector3.ZERO, INK, menu)
	label3(title, Vector3(0, height / 2 - 0.55, 0.05), 64, GOLD, menu)
	if not subtitle.is_empty():
		label3(subtitle, Vector3(0, height / 2 - 1.1, 0.05), 27, CREAM, menu)
	var back := label3(reverse if reverse != "" else title, Vector3(0, 0.2, -0.06), 52, GOLD, menu)
	back.rotation.y = 0.0 if mirrored else PI

func world_button(text: String, y: float, action: String, value: int = 0) -> void:
	mesh_box(Vector3(4.6, 0.48, 0.035), Vector3(0, y, 0.08), Color("594334"), menu)
	label3(text, Vector3(0, y, 0.105), 30, CREAM, menu)
	menu_buttons.append({"y": y, "action": action, "value": value})

func show_title() -> void:
	panel("KOI KOI", word("subtitle"), "KOI KOI")
	world_button("START", -0.35, "start")
	world_button(word("settings"), -0.98, "settings")
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
		left_label.text = "KOI  /  KOI\n" + word("title")
		right_label.text = "12  /  " + word("round")
		bottom_label.text = "THE FLOATING TABLE"
		return
	var pts: Array = state.points
	left_label.text = "%s    %02d / 12\n%s  %d %s%s" % [word("round"), state.round, word("you"), pts[0], word("points"), "  · " + word("dealer") if state.dealer == 0 else ""]
	right_label.text = "%s  %d %s%s\n%s" % [word("ai"), pts[1], word("points"), "  · " + word("dealer") if state.dealer == 1 else "", word("turn_you") if state.player == 0 else word("turn_ai")]
	bottom_label.text = word("thinking") if busy else word("pick") if state.phase in [1, 2] else word("play")
	if paused:
		bottom_label.text = word("final") if finished else "%s %d / 12" % [word("round"), state.round]
		right_label.text = right_label.text.get_slice("\n", 0) + "\n" + word("final" if finished else "round_complete")

func send(path: String, data: Dictionary = {}) -> void:
	if busy:
		return
	busy = true
	inspector.hide()
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
		if int(state.player) == 0:
			world_button(word("yes"), -0.35, "action", 49)
			world_button(word("no"), -0.98, "action", 48)
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
	var height := maxf(4.2, 2.8 + (ceilf(lines.size() / 2.0) if columns else lines.size()) * 0.36)
	panel(headline, word("final") if finished else "%s %d / 12" % [word("round"), state.round], "LOSE" if headline == "WIN" else "WIN" if headline == "LOSE" else "DRAW", height)
	if columns:
		var middle := ceili(lines.size() / 2.0)
		label3("\n".join(lines.slice(0, middle)), Vector3(-1.8, -0.15, 0.06), 23, CREAM, menu)
		label3("\n".join(lines.slice(middle)), Vector3(1.8, -0.15, 0.06), 23, CREAM, menu)
	else:
		label3(text, Vector3(0, -0.15, 0.06), 25, CREAM, menu)
	world_button(word("new" if finished else "next"), -height / 2.0 + 0.4, "start" if finished else "next")

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
	card.identity = id
	if id >= 0:
		card.set_face(textures[id])
		known[id] = card
	return card

func hand_position(slot: int, player: int) -> Vector3:
	return Vector3((slot - 3.5) * 0.87, 1.25, 3.85 if player == 0 else -3.85)

func field_position(card_id: int) -> Vector3:
	if not field_slots.has(card_id):
		var occupied := field_slots.values()
		for i in range(32):
			if i not in occupied:
				field_slots[card_id] = i
				break
	var index: int = field_slots[card_id]
	if index < 8:
		return Vector3((index % 4 - 1.5) * 1.1, 0.075, (index / 4 - 0.5) * 1.35)
	if index < 16:
		return Vector3((index % 4 - 1.5) * 1.1, 0.075, -2.0 if index < 12 else 2.0)
	return Vector3(-3.5 - ((index - 16) / 4) * 0.79, 0.075, ((index - 16) % 4 - 1.5) * 1.3)

func fly(card, pos: Vector3, rotation: Vector3, seconds: float = 0.7, lift: float = 0.6, circle: bool = false) -> void:
	card.move_to(pos, rotation, seconds / speed, lift, circle)

func reset_deck() -> void:
	clear_yaku_labels()
	known.clear()
	opponent_cards.clear()
	field_slots.clear()
	stack.clear()
	# Return in ordered waves. Every card keeps its transform until its flight starts.
	for i in range(bodies.size()):
		var card = bodies[i]
		fly(card, STACK + Vector3(0, i * 0.047, 0), BACK_FLAT, 0.9, 1.2 + (i % 4) * 0.09)
		stack.append(card)
		await wait_for(0.012)
	await wait_for(1.0)
	for card in bodies:
		card.identity = -1
		card.set_face(null)

func opening(event: Dictionary) -> void:
	await reset_deck()
	for reveal in event.reveals:
		panel(word("opening"), "", word("opening"), 1.7)
		menu.position.y = 3.5
		var pair: Array = []
		for p in range(2):
			var card = take_card(int(reveal.cards[p]))
			pair.append(card)
			fly(card, Vector3(-0.65 if p == 0 else 0.65, 1.7, 0), Vector3(0, PI * p, 0), 1.0)
		await wait_for(1.1)
		for card in pair:
			fly(card, card.position, card.rotation + Vector3(0, PI, 0), 1.5, 0)
		await wait_for(1.6)
		for card in pair:
			fly(card, card.position, card.rotation + Vector3(0, PI, 0), 1.5, 0)
		await wait_for(1.5)
		panel(word("tie") if int(reveal.winner) < 0 else word("dealer_you" if int(reveal.winner) == 0 else "dealer_ai"), "", "", 1.7)
		menu.position.y = 3.5
		await wait_for(1.0)
		await reset_deck()
	clear_menu()
	await deal_cards(event.view, false)

func deal_cards(view: Dictionary, returning: bool = true) -> void:
	if returning:
		await reset_deck()
	clear_menu()
	state = view
	refresh_hud()
	# Four sets: field pair, dealer hand pair, other hand pair.
	for batch in range(4):
		for group in [2, int(view.dealer), 1 - int(view.dealer)]:
			for j in range(2):
				var slot := batch * 2 + j
				var id := int(view.field[slot]) if group == 2 else int(view.hand[slot]) if group == 0 else -1
				var card = take_card(id)
				if group == 2:
					fly(card, field_position(id), FLAT)
				else:
					card.slot = slot
					card.seat = group
					if group == 1:
						opponent_cards[slot] = card
					fly(card, hand_position(slot, group), Vector3(0, PI * group, 0))
				await wait_for(0.065)
			await wait_for(0.14)
	await wait_for(0.8)
	tap()

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
		fly(card, card.position, card.rotation + Vector3(0, PI, 0), 0.8, 0)
		await wait_for(0.85)
		# Keep the revealed card aloft until capture/place tells us its destination.
		fly(card, Vector3(0, 1.65, 0), Vector3(0, 0, 0), 0.65, 0.35)
		await wait_for(0.7)
	elif kind == "draw":
		var card = take_card(int(event.card))
		# The sole intentional instantaneous flip specified in the design.
		card.rotation = FLAT
		await wait_for(0.25)
		fly(card, Vector3(0, 1.65, 0), Vector3.ZERO, 0.65, 0.5)
		await wait_for(0.7)
	elif kind in ["capture", "place"]:
		var card = known[int(event.card)]
		var target: Vector3
		if kind == "capture":
			target = known[int(event.targets[0])].position + Vector3(0.14, 0.08, 0.14)
		else:
			target = field_position(int(event.card))
		fly(card, target + Vector3(0, 0.45 if event.new_yaku else 0, 0), FLAT, 0.6, 0.3)
		await wait_for(0.65)
		if event.new_yaku:
			card.strike()
			# Give the actual physics contact time even at fast presentation speed.
			await get_tree().create_timer(0.32).timeout
		tap(1.35 if event.new_yaku else 1.0)
		if kind == "capture":
			for c in event.targets:
				field_slots.erase(int(c))
			await wait_for(0.18)
			await arrange_captures(view)
	elif kind == "koi":
		panel("KOI KOI", word("koi_you" if int(event.player) == 0 else "koi_ai"), "KOI KOI", 3.3, true)
		# Notice faces the recipient; the other player sees the reversed front.
		if int(event.player) == 0:
			menu.rotation.y = PI
		await wait_for(1.4)
		clear_menu()
	elif kind == "round_end":
		if event.total_yaku:
			await sweep_yaku(view, int(event.winner))
			await screenshot("total_yaku")
	state = view
	refresh_hud()

func restore_view(view: Dictionary) -> void:
	# Recovery is a visible return/re-layout, never a replay of an accepted action.
	await reset_deck()
	for slot in range(8):
		var id := int(view.hand[slot])
		if id >= 0:
			var card = take_card(id)
			card.slot = slot
			card.seat = 0
			fly(card, hand_position(slot, 0), Vector3.ZERO)
	for raw_slot in view.opponent_slots:
		var slot := int(raw_slot)
		var card = take_card()
		card.slot = slot
		card.seat = 1
		opponent_cards[slot] = card
		fly(card, hand_position(slot, 1), Vector3(0, PI, 0))
	for raw_id in view.field:
		var id := int(raw_id)
		fly(take_card(id), field_position(id), FLAT)
	for capture in view.captured:
		for raw_id in capture:
			take_card(int(raw_id))
	if int(view.pending) >= 0:
		fly(take_card(int(view.pending)), Vector3(0, 1.65, 0), Vector3.ZERO)
	await arrange_captures(view)
	await wait_for(0.8)
	if paused and ending.get("total_yaku", false):
		await sweep_yaku(view, int(ending.winner))

func clear_yaku_labels() -> void:
	for node in label_nodes:
		node.queue_free()
	label_nodes.clear()

func arrange_captures(view: Dictionary) -> void:
	clear_yaku_labels()
	var changed := false
	for player in range(2):
		var allocated: Array = []
		var row_index := 0
		for row in view.yaku[player]:
			if row.cards.is_empty():
				continue
			var cards_here: Array = []
			for raw_id in row.cards:
				var id := int(raw_id)
				if id not in allocated:
					allocated.append(id)
					cards_here.append(id)
			var z := 5.0 if player == 0 else -5.0
			var y := 2.0 - row_index * 1.32
			var title := label3(word(row.key) + "  " + str(row.points), Vector3(0, y + 0.56, z), 20, INK)
			title.rotation.y = PI if player == 0 else 0.0
			label_nodes.append(title)
			for i in range(cards_here.size()):
				var card = known[cards_here[i]]
				var spacing := minf(0.71, 6.5 / maxf(1, cards_here.size() - 1))
				var x := (3.25 - i * spacing) * (1 if player == 0 else -1)
				var pos := Vector3(x, y - 0.18, z + (0.03 * row_index if player == 0 else -0.03 * row_index))
				# Real cards occupy just one role row; overlapping roles share membership.
				if card.position.distance_to(pos) > 0.015:
					fly(card, pos, Vector3(0, PI if player == 0 else 0.0, 0), 0.85, 0.9)
					changed = true
			row_index += 1
		var index := 0
		for raw_id in view.captured[player]:
			var id := int(raw_id)
			if id in allocated:
				continue
			var pos := Vector3(-4.8 + index * 0.31, 0.08 + index * 0.002, 3.08 if player == 0 else -3.08)
			var card = known[id]
			if card.position.distance_to(pos) > 0.015:
				fly(card, pos, FLAT if player == 0 else Vector3(-PI / 2, 0, PI), 0.75, 0.75)
				changed = true
			index += 1
	if changed:
		await wait_for(0.9)

func sweep_yaku(view: Dictionary, winner: int) -> void:
	clear_yaku_labels()
	var ids: Array = []
	for player in range(2):
		for row in view.yaku[player]:
			for c in row.cards:
				if int(c) not in ids:
					ids.append(int(c))
	for i in range(ids.size()):
		var target := Vector3((i % 12 - 5.5) * 0.7, 0.18 + (i / 12) * 0.05, 3.05 if winner == 0 else -3.05)
		fly(known[ids[i]], target, FLAT, 1.5, 1.3, true)
		await wait_for(0.025)
	await wait_for(1.6)

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
	if modal or busy:
		return
	if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
		if click_menu(event.position):
			return
		if started and not paused and int(state.player) == 0:
			var card = hit_card(event.position)
			if card != null and state.legal.any(func(a): return int(a) == card.identity):
				send("/action", {"action": card.identity})
	if event is InputEventMouseMotion:
		var card = hit_card(event.position)
		var id: int = card.identity if card != null else -1
		if id >= 0 and id != last_hover:
			inspect_image.texture = textures[id]
			inspect_text.text = "%d月 · %s" % [id / 4 + 1, JP_MONTHS[id / 4]] if language == 0 else MONTHS[id / 4].to_upper()
			inspector.show()
		elif id < 0:
			inspector.hide()
		last_hover = id

func hit_card(screen: Vector2):
	var ray := PhysicsRayQueryParameters3D.create(camera.project_ray_origin(screen), camera.project_ray_origin(screen) + camera.project_ray_normal(screen) * 80.0, 2)
	var hit := get_world_3d().direct_space_state.intersect_ray(ray)
	return hit.get("collider")

func click_menu(screen: Vector2) -> bool:
	if not is_instance_valid(menu) or menu_buttons.is_empty():
		return false
	var ray := camera.project_ray_normal(screen)
	var origin := camera.project_ray_origin(screen)
	var plane := Plane(menu.global_basis.z, menu.global_position + menu.global_basis.z * 0.12)
	var point = plane.intersects_ray(origin, ray)
	if point == null:
		return false
	var local: Vector3 = menu.to_local(point)
	for item in menu_buttons:
		if absf(local.x) <= 2.3 and absf(local.y - item.y) <= 0.24:
			match item.action:
				"start": send("/start", {"opponent": opponent, "flags": flags})
				"action": send("/action", {"action": item.value})
				"next": send("/next")
				"settings": open_overlay("settings")
				"retry": send("/state")
			return true
	return absf(local.x) < 3.75 and absf(local.y) < 1.65

func open_overlay(kind: String) -> void:
	modal = true
	inspector.hide()
	if is_instance_valid(overlay):
		overlay.queue_free()
	overlay = PanelContainer.new()
	overlay.add_theme_stylebox_override("panel", style(INK, GOLD, 8))
	overlay.set_anchors_and_offsets_preset(Control.PRESET_CENTER)
	overlay.position = Vector2(-385, -365)
	overlay.custom_minimum_size = Vector2(770, 730)
	hud.add_child(overlay)
	var scroll := ScrollContainer.new()
	scroll.custom_minimum_size = Vector2(730, 690)
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
		var credits := ui_label("HANAFUDA\nLouie Mantia · recolored by dotty-dev\nCC BY-SA 4.0\n\ngithub.com/dotty-dev/Hanafuda-Louie-Recolor\nOriginal SVG artwork bundled without modification.\n\nNoto Sans JP · The Noto Project\nSIL Open Font License 1.1\n\nGodot Engine 4.7.2 · MIT\ngodotengine.org\n\nRules & AI integration\nKoiKoi-AI · Sanghai Guan · MIT\n\nFull notices: game/assets and THIRD_PARTY_NOTICES.md", 17)
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
	if is_instance_valid(overlay):
		overlay.queue_free()
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
