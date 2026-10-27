extends SceneTree
## Real input routing through the rendered application, with the local service.
var table
var audit
var failures: Array[String] = []

func _initialize() -> void:
	call_deferred("run")

func check(ok: bool, message: String) -> void:
	if not ok:
		failures.append(message)
		push_error(message)

func click_at(point: Vector2) -> void:
	var motion := InputEventMouseMotion.new()
	motion.position = point
	Input.parse_input_event(motion)
	await process_frame
	var click := InputEventMouseButton.new()
	click.button_index = MOUSE_BUTTON_LEFT
	click.position = point
	click.pressed = true
	Input.parse_input_event(click)
	await process_frame
	click.pressed = false
	Input.parse_input_event(click)
	await process_frame

func until_ready() -> void:
	for i in range(2000):
		await create_timer(0.02).timeout
		if not table.busy:
			return
	check(false, "Game request/animation timed out")

func run() -> void:
	table = load("res://main.tscn").instantiate()
	root.add_child(table)
	audit = load("res://tests/motion_audit.gd").new()
	physics_frame.connect(func(): audit.inspect(table))
	await create_timer(0.4).timeout
	table.speed = 10.0
	table.qa = true
	table.language = 0
	var saved_flags: int = table.flags
	table.flags = 24
	await table.screenshot("title")
	var start: Vector2 = table.camera.unproject_position(table.menu.to_global(Vector3(0, 0, 0.013)))
	await click_at(start)
	check(table.busy, "START did not route through world-space button hit testing")
	await until_ready()
	check(table.started, "START did not create a match")
	if not table.started:
		quit(1)
		return
	# AI may still be scheduled after animation; wait for the human decision.
	for i in range(500):
		if not table.busy and table.state.player == 0:
			break
		await create_timer(0.02).timeout
	check(table.state.player == 0, "AI did not hand control back")
	await table.screenshot("table")
	var action: int = int(table.state.legal[0])
	var card = table.known[action]
	var mouse: Vector2 = table.camera.unproject_position(card.global_position)
	check(table.hit_card(mouse) == card, "Camera ray did not pick the human card")
	var rev: int = table.revision
	await click_at(mouse)
	check(table.busy, "Hand click did not submit an action")
	await until_ready()
	check(table.revision > rev, "Hand action did not reach the backend")
	# Stale/recovery read re-lays the current public view without replaying a move.
	await until_ready()
	table.send("/state")
	await until_ready()
	check(table.bodies.size() == 48, "Recovery changed physical card count")
	table.open_overlay("settings")
	await process_frame
	await table.screenshot("settings")
	check(table.overlay.get_global_rect().position.x >= 0, "Settings panel is off-screen")
	# Click the real language Button, using its observed layout rectangle.
	var language_button = table.overlay_box.get_child(1)
	await click_at(table.overlay_world.screen_point(language_button.get_global_rect().get_center(), table.camera))
	check(table.language == 1, "Language button did not change to English")
	await table.screenshot("settings_en")
	table.open_overlay("help")
	await process_frame
	await table.screenshot("help_en")
	table.close_overlay()
	# A full match through the same rendered event stream covers circular settlement
	# and return/deal transitions with both optional rules enabled.
	table.speed = 12.0
	var checked_koi := false
	var checked_end := false
	var last_revision := -1
	var unchanged := 0
	var last_round := 0
	for i in range(15000):
		await create_timer(0.02).timeout
		if table.revision == last_revision:
			unchanged += 1
		else:
			last_revision = table.revision
			unchanged = 0
		if int(table.state.round) != last_round:
			last_round = int(table.state.round)
			print("GAME_UI_ROUND %d revision=%d" % [last_round, table.revision])
		if unchanged > 700:
			print("GAME_UI_STALL busy=%s paused=%s phase=%s player=%s legal=%s" % [table.busy, table.paused, table.state.phase, table.state.player, table.state.legal])
			await table.screenshot("stall")
			break
		if table.busy:
			continue
		if table.finished:
			break
		if table.paused:
			if not checked_end:
				await table.screenshot("result_ui")
				checked_end = true
			var button: Dictionary = table.menu_buttons[0]
			await click_at(table.camera.unproject_position(table.menu.to_global(Vector3(button.x, button.y, 0.013))))
		elif int(table.state.player) == 0:
			if int(table.state.phase) == 3 and not checked_koi:
				await table.screenshot("human_koi")
				await click_at(table.camera.unproject_position(table.menu.to_global(Vector3(-1.4, -1.35, 0.013))))
				checked_koi = true
			else:
				var pick: int = int(table.state.legal[0])
				if pick == 48:
					await click_at(table.camera.unproject_position(table.menu.to_global(Vector3(1.4, -1.35, 0.013))))
				else:
					await click_at(table.camera.unproject_position(table.known[pick].global_position))
	check(audit.overlaps == 0, "Moving cards intersected")
	check(audit.board_hits == 0, "Moving cards crossed the board")
	check(table.finished, "Full rendered match did not finish")
	check(int(table.state.round) == 12, "Rendered match did not complete twelve rounds")
	await table.screenshot("final_ui")
	# Restore settings to the requested initial language without altering opponent.
	table.language = 0
	table.speed = 1.0
	table.flags = saved_flags
	table.save_settings()
	print("GAME_MOTION_AUDIT max_lift=%.3f normal_max=%.3f overlaps=%d board=%d" % [audit.max_lift, audit.normal_max, audit.overlaps, audit.board_hits])
	print("GAME_UI_CHECKS failures=%d revision=%d" % [failures.size(), table.revision])
	quit(0 if failures.is_empty() else 1)
