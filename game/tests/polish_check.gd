extends SceneTree
var failures: Array[String] = []
func _initialize() -> void: call_deferred("run")
func check(ok: bool, message: String) -> void:
	if not ok and message not in failures:
		failures.append(message)
		push_error(message)
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.qa = true
	table.speed = 3
	await create_timer(0.5).timeout
	check(table.menu_buttons.size()==1 and table.menu_buttons[0].caption.text=="PLAY" and table.menu.has_node("TitleRules"),"Title lacks rules above PLAY")
	await table.screenshot("polish_title_ja")
	table.language = 1
	table.show_title()
	await create_timer(0.6).timeout
	await table.screenshot("polish_title_en")
	var before_mask: int = table.camera.cull_mask
	await table.open_overlay("settings")
	await create_timer(0.3).timeout
	check(table.camera.cull_mask&4==0 and table.overlay_world.surface.layers==8,"Modal did not hide the other world UI")
	var before_camera: Transform3D = table.camera.transform
	var mouse := InputEventMouseButton.new()
	mouse.button_index = MOUSE_BUTTON_RIGHT
	mouse.pressed = true
	mouse.position = Vector2(720,400)
	table._unhandled_input(mouse)
	var motion := InputEventMouseMotion.new()
	motion.position = mouse.position
	motion.relative = Vector2(0,50)
	table._unhandled_input(motion)
	mouse.pressed = false
	table._unhandled_input(mouse)
	check(not table.camera.transform.is_equal_approx(before_camera),"Modal blocks camera orbit")
	table.update_camera()
	for option in table.overlay_box.get_children().filter(func(c):return c is OptionButton or c is CheckButton):
		check(is_equal_approx(option.get_theme_stylebox("normal").content_margin_left,option.get_theme_stylebox("hover").content_margin_left),"Hover shifts the text horizontally")
	await table.screenshot("polish_settings_en")
	var dropdown: OptionButton = table.overlay_box.find_child("MatchRounds",true,false)
	dropdown.show_popup()
	await create_timer(0.15).timeout
	check(dropdown.get_popup().visible,"Styled dropdown did not open")
	await table.screenshot("polish_dropdown_en")
	dropdown.get_popup().hide()
	await table.open_overlay("help")
	var back: Array = table.overlay_box.get_child(0).get_children().filter(func(c):return c is Button and c.text==table.word("back"))
	check(back.size()==1,"Settings subpage lacks BACK")
	back[0].pressed.emit()
	await process_frame
	check(table.overlay_page=="settings","BACK did not return to settings")
	await table.close_overlay()
	check(table.camera.cull_mask==before_mask,"Closing modal did not restore world UI")
	var dense: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/dense_result.json"))
	table.state = dense.view
	table.ending = dense.end
	table.state.points = [50,10]
	table.started = true
	table.finished = true
	table.paused = true
	table.show_result()
	check(table.menu_buttons.any(func(b):return b.action=="home") and table.menu_buttons.any(func(b):return b.action=="start"),"Finished match lacks replay/title choices")
	var plate: MeshInstance3D = table.menu.get_child(0)
	var plate_color: Color = plate.material_override.albedo_color
	table.sound_log.clear()
	await create_timer(0.75).timeout
	check(table.blink_labels[0].modulate.a<0.5 and plate.material_override.albedo_color==plate_color,"WIN blinking affected the background or did not blink")
	await create_timer(0.75).timeout
	check(table.sound_log.any(func(e):return e.sound=="win-tick"),"WIN blinking has no gentle tick")
	await table.screenshot("polish_result_en")
	await table.return_home()
	check(not table.started and table.state.is_empty() and table.stack.size()==48 and table.menu_buttons[0].caption.text=="PLAY","Title return left the old match active")
	# Two independent new roles finish in order, with trails and contact flashes.
	table.language = 0
	table.clear_menu()
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	view.hand = [-1,-1,-1,-1,-1,-1,-1,-1]
	view.opponent_slots = []
	view.field = []
	view.captured = [[0,8,28,1,5,9],[]]
	view.yaku = [[{"key":"three","points":5,"cards":[0,8,28]},{"key":"red","points":5,"cards":[1,5,9]}],[]]
	table.state = view
	table.started = true
	for i in range(6):
		var card = table.take_card(view.captured[0][i])
		card.position = Vector3(-3+i,0.012,0)
		card.rotation = table.FLAT
	var contacts := {}
	var contact := func(c):contacts[c.identity] = Time.get_ticks_msec()
	for c in table.bodies: c.role_hit.connect(contact)
	var arrivals := {}
	var observe := func(c):arrivals[c.identity] = Time.get_ticks_msec()
	for c in table.bodies: c.arrived.connect(observe)
	table.saved_camera_transform = table.camera.transform
	table.camera_focus_armed = true
	table.arrange_captures(view)
	while table.celebration.impacts==0: await physics_frame
	await table.screenshot("polish_role_impact")
	await table.finish_flights()
	check(contacts[1]-maxi(arrivals[0],maxi(arrivals[8],arrivals[28]))>=600,"Second role skipped the readable beat after the first role finished")
	check(table.celebration.impacts>=6 and table.celebration.emitted>960,"Role arrival lacks per-card impacts and growing spark trails")
	check(table.label_nodes.all(func(p):return p.visible) and table.camera.cull_mask==before_mask,"Role view hid its names or changed scene visibility")
	await table.screenshot("polish_roles")
	await table.finish_role_camera()
	for c in table.bodies:
		c.arrived.disconnect(observe)
		c.role_hit.disconnect(contact)
	await table.reset_deck()
	# Both seats return simultaneously; every individual arrival has one cue.
	for i in range(8):
		var c = table.take_card(i)
		c.position = Vector3(-1+(i%4)*0.88,3.55,12.2 if i<4 else -12.2)
		c.rotation = Vector3(0,PI if i<4 else 0,0)
	var simultaneous := [false]
	var watch := func():
		var moving: Array = table.bodies.filter(func(c):return c.flight and c.returning and c.elapsed>c.start_delay)
		simultaneous[0] = simultaneous[0] or (moving.any(func(c):return c.flight_from.z>0) and moving.any(func(c):return c.flight_from.z<0))
	physics_frame.connect(watch)
	table.sound_log.clear()
	await table.reset_deck()
	physics_frame.disconnect(watch)
	check(simultaneous[0] and table.sound_log.filter(func(e):return e.sound=="fuda-turn" and e.card>=0).size()==8,"Return serialized seats or lost individual sounds")
	# The ordinary pickup visibly pauses after the field landing.
	var source = table.take_card(0)
	source.position = Vector3(0,1,3.9)
	var target = table.take_card(1)
	target.position = Vector3(0,0.012,0)
	target.rotation = table.FLAT
	view.captured = [[0,1],[]]
	view.yaku = [[],[]]
	table.sound_log.clear()
	await table.animate_event({"kind":"capture","card":0,"targets":[1],"player":0,"new_yaku":false,"view":view})
	var landing: Array = table.sound_log.filter(func(e):return e.sound=="bafuda-place")
	var pickup: Array = table.sound_log.filter(func(e):return e.sound=="fuda-turn")
	check(landing.size()==1 and not pickup.is_empty() and pickup[0].time-landing[0].time>=320,"Ordinary capture skipped its readable pause")
	print("GAME_POLISH_CHECK failures=%d sparks=%d impacts=%d"%[failures.size(),table.celebration.emitted,table.celebration.impacts])
	quit(0 if failures.is_empty() else 1)
