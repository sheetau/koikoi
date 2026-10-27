extends SceneTree
var failures: Array[String] = []
func _initialize() -> void:
	call_deferred("run")
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
	table.speed = 5
	var fixture: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json"))
	table.state = fixture.view
	table.started = true
	table.refresh_hud()
	var fixed: Array = [table.left_label.transform,table.right_label.transform]
	for zoom in [9.5,18.0,30.0]:
		table.camera_radius = zoom
		table.update_camera()
		table.refresh_hud()
		check(table.left_label.transform.is_equal_approx(fixed[0]) and table.right_label.transform.is_equal_approx(fixed[1]),"Zoom moved world HUD anchors")
	table.camera_radius = 12.6
	table.update_camera()
	# Paused AI phase-three results must still provide hover feedback.
	table.paused = true
	table.state.player = 1
	table.state.phase = 3
	table.ending = {"winner":1,"payment":5,"reason":"stop"}
	table.show_result()
	var button: Dictionary = table.menu_buttons[0]
	table.hover_menu(table.camera.unproject_position(table.menu.to_global(Vector3(button.x,button.y,0.013))))
	check(button.plate.material_override.albedo_color!=button.fill,"Next-round hover is disabled after an AI finish")
	table.announce("短い通知",0.1)
	var label: Label = table.message_world.viewport.get_child(0).get_child(0).get_child(0)
	check(label.horizontal_alignment==HORIZONTAL_ALIGNMENT_CENTER,"Short message is not centered")
	await create_timer(1.05).timeout
	# All 48 distinct cards, across both hands, field, captures and role rows.
	table.paused = false
	table.clear_menu()
	for id in range(48):
		var card = table.take_card(id)
		if id<16:
			card.position = table.hand_position(id%8,id/8)
			card.rotation = Vector3(0,PI*(id/8),0)
		elif id<32:
			card.position = table.Layout.field(id-16,16)
			card.rotation = table.FLAT
		elif id<40:
			card.position = Vector3(-3.4+(id-32)*0.9,0.012,5.15)
			card.rotation = table.FLAT
		else:
			var side := 1 if id<44 else -1
			card.position = Vector3((3-(id%4)*0.88)*side,3.55,12.2*side)
			card.rotation = Vector3(0,PI if side==1 else 0,0)
	var audit = load("res://tests/motion_audit.gd").new()
	var count := [0]
	var observe := func():
		audit.inspect(table)
		count[0] = maxi(count[0],table.bodies.filter(func(c):return c.flight and c.returning and c.elapsed>c.start_delay).size())
		for card in table.bodies:
			if card.flight and card.returning:
				check(card.visible and card.gather_group==-1,"Return merged or hid a traveling card")
				if card.bottom_return:
					var line: Vector2 = Vector2(card.flight_to.x-card.flight_from.x,card.flight_to.z-card.flight_from.z)
					var offset: Vector2 = Vector2(card.position.x-card.flight_from.x,card.position.z-card.flight_from.z)
					check(absf(line.cross(offset))<0.001,"Horizontal return does not follow a straight line")
	physics_frame.connect(observe)
	await table.reset_deck()
	physics_frame.disconnect(observe)
	check(table.stack.size()==48 and count[0]>1,"Convoy lost cards or never formed a line")
	check(audit.overlaps==0 and audit.board_hits==0,"Convoy intersects another card or the board")
	var pending = table.take_card(47)
	pending.rotation = table.FLAT
	await table.reset_deck()
	check(table.stack.size()==48 and not pending.visible and not pending.flight,"Pending stock-top card took an unnecessary return loop")
	# Height interpolates continuously throughout total-yaku motion.
	var sample = table.take_card(0)
	sample.position = Vector3(-3,3.55,-12.2)
	sample.rotation = Vector3.ZERO
	table.fly(sample,Vector3(3,0.51,12.2),Vector3(0,PI,0),1.2,0,true)
	table.plan_flights()
	for i in range(101):
		var u := i/100.0
		var expected := lerpf(3.55,0.51,u*u*(3-2*u))
		check(absf(sample.pose_at(sample.duration*u).origin.y-expected)<0.001,"Transfer clearance overrides interpolated destination height")
	await table.finish_flights()
	await table.reset_deck()
	# Separate contact-driven bodies vary their angles and landing offsets.
	var landings: Array = []
	for attempt in range(3):
		var base = table.take_card(0)
		base.position = Vector3(0,0.012,0)
		base.rotation = table.FLAT
		var incoming = table.take_card(8)
		incoming.position = Vector3(0.13,0.512,0.13)
		incoming.rotation = table.FLAT
		await table.impact_capture(incoming,[0])
		check(table.last_impact_settled and table.last_burst_peak>0.25,"Impact lost its original burst or failed to settle")
		check(table.impact_samples.any(func(frame):return frame[0].angles.distance_to(frame[1].angles)>0.02),"Impacted cards keep identical angles")
		var rebounds := 0
		for i in range(1,table.impact_samples.size()):
			if table.impact_samples[i-1][0].velocity.y < -0.05 and table.impact_samples[i][0].velocity.y > 0.05:
				rebounds += 1
		check(rebounds>=2,"Impact lacks the smaller secondary bounce")
		landings.append(incoming.position)
		await table.reset_deck()
	check(landings[0].distance_to(landings[1])>0.005,"Impact landing does not vary")
	# Observe the exact U-turn boundary and restore the player's transform.
	var view: Dictionary = fixture.view.duplicate(true)
	view.captured = [[0,8,28],[]]
	view.yaku = [[{"key":"three","points":5,"cards":[0,8,28]}],[]]
	table.state = view
	for i in range(3):
		var card = table.take_card([0,8,28][i])
		card.position = Vector3(-2+i,0.012,0)
		card.rotation = table.FLAT
	var before: Transform3D = table.camera.transform
	table.saved_camera_transform = before
	table.camera_focus_armed = true
	table.arrange_captures(view)
	while not table.role_camera_active and table.bodies.any(func(c):return c.flight):
		check(table.camera.transform.is_equal_approx(before),"Camera moved before the U-turn")
		await physics_frame
	check(table.role_camera_active,"Camera missed the role U-turn")
	await table.screenshot("flow_role_camera")
	await table.finish_flights()
	table.refresh_owned_roles(view)
	await table.screenshot("flow_role_arrived")
	var held_at := Time.get_ticks_msec()
	await table.finish_role_camera()
	check(Time.get_ticks_msec()-held_at>=440 and table.camera.transform.is_equal_approx(before),"Camera did not hold then restore the original view")
	table.refresh_hud()
	await table.screenshot("flow_role_list")
	print("GAME_FLOW_CHECK failures=%d convoy_max=%d overlaps=%d board=%d landings=%s"%[failures.size(),count[0],audit.overlaps,audit.board_hits,landings])
	quit(0 if failures.is_empty() else 1)
