extends SceneTree
## Public fixtures check audio timing and the explicitly requested visual merge.
var errors: Array[String] = []
func _initialize() -> void:
	call_deferred("run")
func check(ok: bool, message: String) -> void:
	if not ok:
		errors.append(message)
		push_error(message)
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.speed = 1
	for key in table.SOUNDS:
		check(table.SOUNDS[key].get_length() > 0, "Empty sound: " + key)
	var opening: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json"))
	await table.opening(opening.events[0])
	check(table.sound_log.filter(func(e): return e.sound == "tefuda-place").size() == 18, "Dealer/hand landing sounds must play once per card")
	check(table.sound_log.filter(func(e): return e.sound == "bafuda-place").size() == 8, "Field landing sounds must play once per card")
	await table.reset_deck()
	var view: Dictionary = opening.view.duplicate(true)
	view.captured = [[0, 8, 28], []]
	view.yaku = [[{"key": "three", "points": 5, "cards": [0, 8, 28]}], []]
	for i in range(3):
		var card = table.take_card([0, 8, 28][i])
		card.position = Vector3(i - 1, 0.012, 0)
		card.rotation = table.FLAT
		card.arrived.connect(func(c):
			if c.role_route:
				check(c.position.distance_to(c.flight_to) < 0.001, "Yaku sound played before arrival"))
	table.sound_log.clear()
	await table.arrange_captures(view)
	check(table.sound_log.filter(func(e): return e.sound == "yaku").size() == 3, "Yaku sounds must play per arriving card")
	var arrivals: Array = table.sound_log.filter(func(e): return e.sound == "yaku").map(func(e): return e.card)
	var cues: Array = table.sound_log.filter(func(e): return e.sound == "yaku")
	for i in range(1, cues.size()):
		check(absf(float(cues[i].time - cues[i - 1].time) - 100.0) < 20, "Role arrival cadence is uneven")
	check(arrivals == [0, 8, 28], "Field-to-role cards did not arrive left first")
	check(table.known.values().all(func(c): return c.role_route), "Field-to-role transfer skipped the outer turn")
	table.sound_log.clear()
	await table.sweep_yaku(view, 1)
	check(table.sound_log.filter(func(e): return e.sound == "fuda-take").size() == 3, "Total-yaku transfer needs a take sound for each card")
	check(table.sound_log.filter(func(e): return e.sound == "yaku").size() == 3, "Total-yaku arrivals need a sound for each card")
	await table.reset_deck()
	var cards: Array = []
	for i in range(3):
		var card = table.take_card(2 + 4 * i)
		card.position = Vector3(-3 + i * 0.39, 0.014 + i * 0.012, 3.65)
		card.rotation = table.FLAT
		cards.append(card)
	var corner: Vector3 = cards[2].position
	var observed := [false, false]
	var inspect := func():
		for card in cards:
			if card.flight and card.gather_group >= 0:
				observed[0] = true
				check(card.flight_to.is_equal_approx(corner), "Group did not gather to its existing outer corner")
				check(card.lift == 0, "Gather added stack height")
			elif card.flight and card.returning:
				observed[1] = true
				check(absf(card.rotation_from.dot(card.rotation_to)) > 0.9999, "Horizontal return packet flipped")
	physics_frame.connect(inspect)
	await table.reset_deck()
	physics_frame.disconnect(inspect)
	check(observed.all(func(value): return value) and table.stack.size() == 48, "Return packet sequence lost cards or skipped a stage")
	print("GAME_AUDIO_RETURN failures=%d clips=%d cards=%d" % [errors.size(), table.SOUNDS.size(), table.stack.size()])
	quit(0 if errors.is_empty() else 1)
