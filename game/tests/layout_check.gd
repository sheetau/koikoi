extends SceneTree
## A synthetic, fully public table exercises dense role/result layout.

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
	table.language = 1
	table.show_decision()
	table.refresh_hud()
	await table.screenshot("dense_result_en")
	table.open_overlay("help")
	await process_frame
	await table.screenshot("help_en")
	print("GAME_LAYOUT_CHECK bodies=%d known=%d" % [table.bodies.size(), table.known.size()])
	quit(0 if table.bodies.size() == 48 and table.known.size() == 48 else 1)
