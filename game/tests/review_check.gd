extends SceneTree
## Public presentation cases: independent count roles, arrival timing and bounded UI.
var failures: Array[String] = []
func _initialize() -> void:
	call_deferred("run")
func check(ok: bool, message: String) -> void:
	if not ok:
		failures.append(message)
		push_error(message)
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.speed = 5
	table.qa = true
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	view.hand = [-1,-1,-1,-1,-1,-1,-1,-1]
	view.opponent_slots = []
	view.field = []
	view.captured = [[28,32,2,3,6,7,10,11,14,15,18],[20,24,36]]
	view.yaku = [[{"key":"moon","points":1,"cards":[28,32]},{"key":"kasu","points":1,"cards":[2,3,6,7,10,11,14,15,18,32]}],[{"key":"animals","points":5,"cards":[20,24,36]}]]
	table.state = view
	table.started = true
	for player in range(2):
		for i in range(view.captured[player].size()):
			var card = table.take_card(view.captured[player][i])
			card.position = Vector3(-4+i*0.75,0.012,1.4 if player==0 else -1.4)
			card.rotation = table.FLAT
	var shown := {}
	var arrived := func(card):
		for plaque in table.label_nodes:
			if plaque.visible and not shown.has(plaque):
				shown[plaque] = card.identity
	for card in table.bodies:
		card.arrived.connect(arrived)
	table.arrange_captures(view)
	check(table.label_nodes.all(func(p):return not p.visible),"Role name appeared before its first card arrived")
	await table.finish_flights()
	await process_frame
	check(table.label_nodes.size()==3 and shown.size()==3,"Count role merged with sake viewing, or title missed arrival")
	check(table.known[28].position.y!=table.known[2].position.y,"Moon viewing and chaff share a row")
	# Both total-yaku directions must enter like ordinary role motion and leave
	# winner transforms and role membership unchanged.
	for winner in range(2):
		await table.restore_view(view)
		var fixed := {}
		for id in view.captured[winner]:
			fixed[id] = table.known[id].transform
		table.sweep_yaku(view,winner)
		for id in view.captured[1-winner]:
			var card = table.known[id]
			if card.flight:
				var before: Vector3 = card.pose_at(card.duration*0.99).origin
				check((card.flight_to.x-before.x)*(1 if winner==0 else -1)>0,"Total-yaku entered from the opposite side")
		await table.finish_flights()
		for id in fixed:
			check(table.known[id].transform.is_equal_approx(fixed[id]),"Total-yaku moved a winner card")
	# A dense result has bounded width/height, can scroll, and grows upward.
	var dense: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/dense_result.json"))
	# The taller result now fits the original fixture; stress overflow explicitly.
	for repeat in range(2):
		dense.view.yaku[0].append_array(dense.view.yaku[0].duplicate(true))
	table.state = dense.view
	table.ending = dense.end
	table.paused = true
	table.finished = true
	table.language = 1
	table.refresh_hud()
	table.show_result()
	await create_timer(0.35).timeout
	check(table.result_world.PIXELS.x<=740 and table.result_world.PIXELS.y<=500,"Result viewport exceeded its bounds")
	check(table.result_world.global_position.y-table.result_world.SIZE.y*table.result_world.global_basis.y.length()/2>0.8,"Result panel crossed the board")
	var header: MeshInstance3D = table.menu.get_child(0)
	var list_top: float = table.result_world.position.y+table.result_world.SIZE.y/2
	check(header.position.y-header.mesh.size.y/2>=list_top+0.24,"WIN/LOSE overlaps the upward-growing list")
	var bar: VScrollBar = table.result_scroll.get_v_scroll_bar()
	check(bar.max_value>bar.page,"Dense result did not expose scrolling")
	var old_scroll: int = table.result_scroll.scroll_vertical
	var wheel := InputEventMouseButton.new()
	wheel.button_index = MOUSE_BUTTON_WHEEL_DOWN
	wheel.pressed = true
	wheel.position = table.result_world.screen_point(Vector2(100,100),table.camera)
	table._unhandled_input(wheel)
	await process_frame
	check(table.result_scroll.scroll_vertical>old_scroll,"Result wheel input did not reach the scroll area")
	for label in [table.left_label,table.right_label]:
		check(label.has_node("TextBackground"),"HUD text has no shared background")
	check(is_equal_approx(table.left_label.get_node("TextBackground").mesh.size.x,table.right_label.get_node("TextBackground").mesh.size.x),"HUD widths differ")
	await table.screenshot("review_result_en")
	table.language = 0
	table.refresh_hud()
	table.show_result()
	await table.screenshot("review_result_ja")
	root.size = Vector2i(960,1000)
	table.show_result()
	await create_timer(0.35).timeout
	for corner in [Vector2(-1,-1),Vector2(1,1)]:
		var point: Vector3 = table.result_world.to_global(Vector3(corner.x*table.result_world.SIZE.x/2,corner.y*table.result_world.SIZE.y/2,0))
		var pixel: Vector2 = table.camera.unproject_position(point)
		check(root.get_visible_rect().has_point(pixel),"Narrow result panel is off-screen")
	await table.screenshot("review_result_narrow")
	root.size = Vector2i(1440,900)
	# AI feedback highlights the selected control before the click cue.
	table.paused = false
	table.finished = false
	table.state.player = 1
	table.state.phase = 3
	table.show_decision()
	var button: Dictionary = table.menu_buttons[0]
	table.sound_log.clear()
	await table.preview_ai_choice([{"kind":"opening"}])
	check(table.sound_log.is_empty() and button.plate.material_override.albedo_color==button.fill,"Restart fabricated an AI decision")
	var began := Time.get_ticks_msec()
	table.preview_ai_choice([{"kind":"koi"}])
	check(button.plate.material_override.albedo_color!=button.fill and table.sound_log.is_empty(),"AI did not hover before clicking")
	await create_timer(0.56).timeout
	print("AI_FEEDBACK began=%d sounds=%s"%[began,table.sound_log])
	check(table.sound_log.any(func(e):return e.sound=="ui-click" and e.time-began>=380),"AI click lacked a perceptible hover pause")
	await table.open_overlay("settings")
	check(table.overlay_world.surface.material_override.transparency==BaseMaterial3D.TRANSPARENCY_ALPHA and table.overlay_world.surface.material_override.cull_mode==BaseMaterial3D.CULL_DISABLED,"Modal does not render translucently from both sides")
	await table.screenshot("review_settings")
	table.camera.position = Vector3(0,7,-12)
	table.camera.look_at(table.overlay_world.position)
	await table.screenshot("review_settings_back")
	print("GAME_REVIEW_CHECK failures=%d"%failures.size())
	quit(0 if failures.is_empty() else 1)
