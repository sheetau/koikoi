extends SceneTree
## Focused public presentation fixtures, not simulated playing-strength evidence.
var table
var errors: Array[String] = []

func _initialize() -> void:
	call_deferred("run")

func check(ok: bool, message: String) -> void:
	if not ok:
		errors.append(message)
		push_error(message)

func frames(count: int = 30) -> void:
	for i in range(count):
		await physics_frame

func run() -> void:
	table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await frames()
	table.qa = true
	table.speed = 3.0
	check(table.latin_font_loaded, "Requested local AntiqueSix font was not loaded")
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	view.field.append(5)
	view.player = 0
	view.phase = 0
	view.legal = view.hand.duplicate()
	view.stock_count = 23
	table.state = view
	table.started = true
	table.clear_menu()
	await table.restore_view(view)
	table.show_decision()
	await frames()
	var selected = table.known[7]
	var origin: Vector3 = selected.position
	var revision: int = table.revision
	table.handle_table_click(table.camera.unproject_position(selected.position))
	await frames()
	check(table.selected_hand == 7, "Two-match hand was not selected locally")
	check(table.revision == revision and not table.busy, "Selection submitted a move before a target was chosen")
	check(Vector2(selected.position.x, selected.position.z).distance_to(Vector2(origin.x, origin.z)) < 0.001, "Selection moved the card away from its hand slot")
	for id in view.hand:
		if int(id) != 7:
			check(absf(table.known[int(id)].position.y - 1.04) < 0.001, "Unselected hand remained raised")
	await table.screenshot("addendum_selected")
	table.handle_table_click(Vector2(30, 30))
	await frames()
	check(table.selected_hand == -1 and table.revision == revision, "Blank-space click did not cancel locally")
	for id in view.hand:
		check((table.known[int(id)].position.y > 1.2) == not table.matching_field(int(id)).is_empty(), "Capture/discard elevation is incorrect")
	# A hand already aligned with the field skips the disclosure turn.
	origin = selected.position
	await table.animate_event({"kind": "play", "card": 7, "slot": 2, "player": 0, "view": view})
	check(selected.position.distance_to(origin) < 0.001 and selected.basis.z.dot(Vector3.BACK) > 0.99, "Aligned human hand was turned or moved")
	# The opposite-facing AI hand still reveals in place.
	var ai_card = table.opponent_cards[0]
	origin = ai_card.position
	var positions: Array[Vector3] = []
	var record := func(): positions.append(ai_card.position)
	physics_frame.connect(record)
	await table.animate_event({"kind": "play", "card": 10, "slot": 0, "player": 1, "view": view})
	physics_frame.disconnect(record)
	check(not positions.is_empty() and positions.all(func(p): return p.distance_to(origin) < 0.001), "AI disclosure was mixed with translation")
	check(ai_card.basis.z.dot(Vector3.BACK) > 0.99, "AI disclosure did not align its facing")
	# A revealed draw with two matches remains on the stock until capture.
	await table.animate_event({"kind": "draw", "card": 6, "player": 0, "view": view})
	origin = table.known[6].position
	await frames()
	check(table.known[6].position.distance_to(origin) < 0.001, "Pending draw moved before choosing a match")
	check(absf(origin.x - table.STACK.x) < 0.001 and absf(origin.z - table.STACK.z) < 0.001, "Pending draw used a staging location")
	# Camera drag orbits a fixed focus; wheel changes only its radius.
	var radius: float = table.camera_radius
	var pitch: float = table.orbit_pitch
	var press := InputEventMouseButton.new()
	press.button_index = MOUSE_BUTTON_LEFT
	press.position = Vector2(50, 50)
	press.pressed = true
	table._unhandled_input(press)
	var motion := InputEventMouseMotion.new()
	motion.position = Vector2(50, 110)
	motion.relative = Vector2(0, 60)
	table._unhandled_input(motion)
	press.pressed = false
	table._unhandled_input(press)
	check(table.orbit_pitch != pitch and absf(table.camera.position.distance_to(table.camera_pivot) - radius) < 0.001, "Drag did not orbit at constant radius")
	press.button_index = MOUSE_BUTTON_WHEEL_UP
	press.pressed = true
	table._unhandled_input(press)
	check(table.camera_radius < radius, "Wheel did not zoom")
	table.orbit_pitch = pitch
	table.camera_radius = radius
	table.update_camera()
	# Cards return in concurrent region groups, without an alignment waypoint.
	var expected: int = table.bodies.filter(func(c): return c.visible).size()
	var counts: Array[int] = []
	var count_flights := func(): counts.append(table.bodies.filter(func(c): return c.flight).size())
	physics_frame.connect(count_flights)
	await table.reset_deck()
	physics_frame.disconnect(count_flights)
	check(not counts.is_empty() and counts.max() > 1, "Return groups did not move together")
	check(table.stack.size() == 48, "Simultaneous return lost cards")
	check(table.equivalent_targets([2, 3]) and table.equivalent_targets([45.0, 46.0, 47.0]), "Identical chaff did not auto-select")
	check(not table.equivalent_targets([4, 5]) and not table.equivalent_targets([2, 6]), "Different captures were auto-selected")
	# Field holes are reused without shifting existing cards or changing rows.
	table.configure_field(range(8))
	var old: Dictionary = {}
	for id in range(8):
		old[id] = table.field_position(id)
	table.configure_field([0, 2, 3, 4, 5, 6, 7])
	for id in [0, 2, 3, 4, 5, 6, 7]:
		check(table.field_position(id).is_equal_approx(old[id]), "Removing a field card shifted the others")
	table.configure_field([0, 2, 3, 4, 5, 6, 7, 8])
	check(table.field_position(8).is_equal_approx(old[1]), "Field hole was not reused")
	table.configure_field(range(9))
	for id in [0, 2, 3, 4, 5, 6, 7]:
		check(is_equal_approx(table.field_position(id).z, old[id].z), "Field expansion changed row ordering")
	# Expansion always has two rows and sufficient full-size card separation.
	for count in [8, 16, 17, 24, 32]:
		var points: Array[Vector3] = []
		for i in range(count):
			points.append(table.Layout.field(i, count))
		check(points.all(func(p): return absf(absf(p.z) - 0.85) < 0.001), "Field added a third row")
		for i in range(count):
			for j in range(i):
				check(absf(points[i].x - points[j].x) >= 0.78 or absf(points[i].z - points[j].z) >= 1.28, "Expanded field overlaps cards")
	view.hand = [-1, -1, -1, -1, -1, -1, -1, -1]
	view.opponent_slots = []
	view.field = range(24)
	view.pending = -1
	view.legal = []
	table.state = view
	await table.restore_view(view)
	table.camera_radius = 17
	table.update_camera()
	await table.screenshot("addendum_field24")
	# Strong landing lifts the incoming card and target visibly under physics.
	var top = table.take_card(40)
	table.fly(top, table.known[0].position + Vector3(0.13, 0.5, 0.13), table.FLAT)
	await table.finish_flights()
	await table.impact_capture(top, [0])
	check(table.last_burst_peak > 0.25, "Strong landing did not lift both cards")
	check(table.last_impact_settled and top.position.y < 0.075 and table.known[0].position.y < 0.075, "Cards were collected before landing")
	await table.screenshot("addendum_impact")
	# The question and both choices stay visible together.
	table.state.phase = 3
	table.show_decision()
	var texts: Array = table.menu.get_children().filter(func(c): return c is Label3D).map(func(c): return c.text)
	check(texts.size() == 3 and "はい" in texts and "いいえ" in texts and texts.any(func(text): return "こいこい" in text), "Koi question and choices were not shown together")
	var button: Dictionary = table.menu_buttons[0]
	table.hover_menu(table.camera.unproject_position(table.menu.to_global(Vector3(button.x, button.y, 0.013))))
	check(button.plate.material_override.albedo_color != button.fill, "World button has no hover style")
	table.hover_menu(Vector2(1, 1))
	check(button.plate.material_override.albedo_color == button.fill, "World button hover did not clear")
	# A card physically behind the 3D panel must not paint over it.
	table.panel("START")
	await process_frame
	RenderingServer.force_draw(false)
	var point: Vector3 = table.menu.to_global(Vector3(0.2, -0.22, 0.008))
	var pixel: Vector2i = Vector2i(table.camera.unproject_position(point))
	var before := root.get_texture().get_image().get_pixelv(pixel)
	var behind = table.take_card(47)
	behind.position = table.camera.position + (point - table.camera.position) * 1.2
	behind.rotation = Vector3.ZERO
	await process_frame
	RenderingServer.force_draw(false)
	var after := root.get_texture().get_image().get_pixelv(pixel)
	check(Vector3(before.r, before.g, before.b).distance_to(Vector3(after.r, after.g, after.b)) < 0.02, "Rear card painted over world UI")
	var started_at := Time.get_ticks_msec()
	await table.announce("AI calls Koikoi", 0.1, 1)
	check(table.announcement.double_sided and table.announcement.basis.z.dot(Vector3.FORWARD) > 0.99, "Announcement did not face its recipient")
	check(Time.get_ticks_msec() - started_at >= 980, "Announcement disappeared before its minimum display time")
	table.announce_outcome(0, true)
	var front = table.get_node("OutcomeFront")
	var back = table.get_node("OutcomeBack")
	check(front.text.contains("勝ち") and back.text.contains("負け"), "Outcome faces were not shown simultaneously")
	check(not front.double_sided and not back.double_sided and front.basis.z.dot(back.basis.z) < -0.99, "Outcome faces bleed through each other")
	await create_timer(1.4).timeout
	# Wide and tall windows render the gold world into all four corners.
	for size in [Vector2i(1600, 700), Vector2i(960, 1000)]:
		root.size = size
		await process_frame
		RenderingServer.force_draw(false)
		var shot := root.get_texture().get_image()
		for p in [Vector2i(1, 1), Vector2i(shot.get_width()-2, 1), Vector2i(1, shot.get_height()-2), Vector2i(shot.get_width()-2, shot.get_height()-2)]:
			var color := shot.get_pixelv(p)
			check(color.r > 0.3 and color.g > 0.2, "Letterbox detected")
		await table.screenshot("addendum_%dx%d" % [size.x, size.y])
	print("GAME_ADDENDUM_CHECK failures=%d returned_cards=%d burst_peak=%.3f" % [errors.size(), expected, table.last_burst_peak])
	quit(0 if errors.is_empty() else 1)
