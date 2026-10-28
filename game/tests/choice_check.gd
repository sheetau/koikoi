extends SceneTree
var failures: Array[String] = []
func _initialize() -> void: call_deferred("run")
func check(ok: bool, message: String) -> void:
	if not ok:
		failures.append(message)
		push_error(message)
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.qa = true
	table.speed = 4
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	view.field.append(5)
	view.player = 0
	view.phase = 0
	view.legal = view.hand.duplicate()
	table.state = view
	table.started = true
	await table.restore_view(view)
	table.show_decision()
	await create_timer(0.25).timeout
	table.handle_table_click(table.camera.unproject_position(table.known[7].position))
	check(table.selected_hand==7 and is_instance_valid(table.pick_message),"Hand selection lacks a field-choice prompt")
	await create_timer(0.3).timeout
	await table.screenshot("choice_hand_ja")
	table.handle_table_click(Vector2(30,30))
	check(table.selected_hand==-1 and table.pick_message==null,"Canceling selection left the prompt visible")
	# A human two-target draw remains interactive without an added pause.
	view.phase = 2
	view.pending = 6
	view.legal = [4,5]
	var began := Time.get_ticks_msec()
	await table.animate_event({"kind":"draw","card":6,"player":0,"view":view})
	var manual_ms := Time.get_ticks_msec()-began
	check(manual_ms<200,"Manual stock choice gained an automatic pause")
	table.show_decision()
	check(is_instance_valid(table.pick_message),"Draw choice lacks the central prompt")
	table.language = 1
	table.refresh_hud()
	check(table.pick_message_text=="CHOOSE A FIELD CARD","Choice prompt did not change language")
	await create_timer(0.3).timeout
	await table.screenshot("choice_stock_en")
	table.busy = true
	table.update_hand_affordance()
	check(table.pick_message==null,"Submitting a choice left the prompt visible")
	table.busy = false
	view.legal = [6,7]
	table.refresh_hud()
	check(table.pick_message==null,"Equivalent automatic targets show a choice prompt")
	view.player = 1
	view.legal = []
	table.refresh_hud()
	check(table.pick_message==null,"AI choice asks the human to select a field card")
	# Zero, one, three, equivalent-two and AI-two targets all pause with
	# the revealed card still at the stock, independent of the speed setting.
	table.started = false
	var timings: Array = []
	for item in [[40,[],0],[6,[4],0],[7,[4,5,6],0],[4,[6,7],0],[6,[4,5],1]]:
		await table.reset_deck()
		view.field = item[1]
		view.player = item[2]
		view.phase = 0
		view.legal = []
		var moved := [false]
		var watch := func():
			if table.known.has(item[0]):
				var card = table.known[item[0]]
				moved[0] = moved[0] or card.flight or absf(card.position.x-table.STACK.x)>0.001 or absf(card.position.z-table.STACK.z)>0.001
		physics_frame.connect(watch)
		began = Time.get_ticks_msec()
		await table.animate_event({"kind":"draw","card":item[0],"player":item[2],"view":view})
		timings.append(Time.get_ticks_msec()-began)
		physics_frame.disconnect(watch)
		check(timings[-1]>=650 and not moved[0],"Automatic draw skipped its stationary reveal beat")
	print("GAME_CHOICE_CHECK failures=%d manual_ms=%d automatic_ms=%s"%[failures.size(),manual_ms,timings])
	quit(0 if failures.is_empty() else 1)
