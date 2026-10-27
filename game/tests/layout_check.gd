extends SceneTree
## A synthetic, fully public table exercises dense role/result layout.
var failures: Array[String] = []

func check(ok: bool, message: String) -> void:
	if not ok:
		failures.append(message)
		push_error(message)

func fits(label: Label3D, plate: MeshInstance3D) -> bool:
	var bounds := label.get_aabb()
	var low := bounds.position + label.position - plate.position
	var high := bounds.end + label.position - plate.position
	var size: Vector2 = plate.mesh.size
	return low.x >= -size.x / 2 - 0.01 and high.x <= size.x / 2 + 0.01 and low.y >= -size.y / 2 - 0.01 and high.y <= size.y / 2 + 0.01

func inspect_text(table) -> void:
	for player in range(2):
		for slots in table.role_slots[player].values():
			var heights: Array = slots.keys().map(func(id): return table.known[id].position.y)
			check(heights.is_empty() or heights.max() - heights.min() < 0.001, "A yaku wrapped onto a second row")
	for plaque in table.label_nodes:
		check(fits(plaque.get_child(1), plaque.get_child(0)), "Yaku name exceeds its plate")
	var plates: Array = table.menu.get_children().filter(func(c): return c is MeshInstance3D)
	for label in table.menu.get_children():
		if label is Label3D and label.rotation.y == 0 and not is_equal_approx(label.position.y, -0.75):
			check(plates.any(func(p): return fits(label, p)), "Result/button text exceeds its plate: " + label.text)


func _initialize() -> void:
	call_deferred("run")

func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await create_timer(0.2).timeout
	table.qa = true
	table.speed = 10.0
	var fixture: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/dense_result.json"))
	table.state = fixture.view
	table.ending = fixture.end
	table.paused = true
	table.finished = true
	table.started = true
	await table.restore_view(fixture.view)
	table.show_decision()
	table.refresh_hud()
	await table.screenshot("dense_result_ja")
	check(table.FONT == table.SERIF, "Japanese UI did not use Noto Serif for all characters")
	inspect_text(table)
	var settled: Dictionary = {}
	for id in table.known:
		settled[id] = table.known[id].transform
	await table.switch_language()
	for id in settled:
		check(table.known[id].transform.is_equal_approx(settled[id]), "Language switch moved a settled card")
	table.show_decision()
	table.refresh_hud()
	await table.screenshot("dense_result_en")
	check(table.FONT == table.latin_font, "English UI did not use Antique Six")
	inspect_text(table)
	table.open_overlay("help")
	await process_frame
	await table.screenshot("help_en")
	print("GAME_LAYOUT_CHECK failures=%d bodies=%d known=%d" % [failures.size(), table.bodies.size(), table.known.size()])
	quit(0 if failures.is_empty() and table.bodies.size() == 48 and table.known.size() == 48 else 1)
