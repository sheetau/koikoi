extends SceneTree
var errors: Array[String] = []
func _initialize() -> void:
	call_deferred("run")
func check(ok: bool, message: String) -> void:
	if not ok and message not in errors:
		errors.append(message)
		push_error(message)
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.speed = 3
	var audit = load("res://tests/motion_audit.gd").new()
	var captures: Array = []
	for id in [2,6,10,4,1,0]:
		var card = table.take_card(id)
		card.position = table.Layout.capture(id, captures.size()%3, 0)
		card.rotation = table.FLAT
		captures.append(card)
	for i in range(4):
		var card = table.take_card(14+i)
		card.position = table.Layout.field(i,8)
		card.rotation = table.FLAT
	var role = table.take_card(28)
	role.position = Vector3(3,3.55,-12.2)
	role.rotation = Vector3.ZERO
	var seen := [false,false,false]
	var check_motion := func():
		audit.inspect(table)
		if captures[0].gather_group>=0:
			seen[0] = true
			check(captures.all(func(c):return c.gather_group==captures[0].gather_group), "Capture kinds were split into separate return packets")
		for card in table.bodies:
			if not card.flight or not card.returning:
				continue
			if card.bottom_return:
				seen[1] = true
				check(is_equal_approx(card.flight_to.y,0.012), "Horizontal packet targeted the stock top")
				check(card.position.y<=card.flight_from.y+0.005, "Horizontal return lifted above its initial plane")
				check(absf(card.rotation_from.dot(card.rotation_to))>0.9999, "Horizontal packet flipped")
				if absf(card.position.x-table.STACK.x)<0.77 and absf(card.position.z-table.STACK.z)<1.27 and not table.stack.is_empty():
					check(card.position.y+0.003<=table.stock_visual.position.y+0.005,"Bottom packet passed through the solid stock")
			elif card==role:
				seen[2] = true
				check(card.flight_to.y>0.1,"Upright role packet was not returned to the top")
	physics_frame.connect(check_motion)
	await table.reset_deck()
	physics_frame.disconnect(check_motion)
	check(seen.all(func(v):return v), "Return fixture did not exercise all stages")
	check(table.stack.size()==48 and table.stock_visual.position.y==0, "Return lost a card or left the deck elevated")
	check(audit.overlaps==0 and audit.board_hits==0,"Grouped bottom return crossed another card/board")
	check(table.sound_log.all(func(e):return e.sound=="fuda-turn"),"Return used the role-movement sound")
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	table.sound_log.clear()
	await table.animate_event({"kind":"draw","card":8,"view":view})
	check(table.sound_log.size()==1 and table.sound_log[0].sound=="fuda-turn", "Draw did not use only fuda-turn")
	await table.reset_deck()
	var opponent = table.take_card(-1)
	opponent.position = table.hand_position(0,1)
	opponent.rotation.y = PI
	table.opponent_cards[0] = opponent
	table.sound_log.clear()
	await table.animate_event({"kind":"play","card":12,"slot":0,"player":1,"view":view})
	check(table.sound_log.is_empty(),"Opponent disclosure was not silent")
	await table.reset_deck()
	var role_view: Dictionary = view.duplicate(true)
	role_view.captured = [[0,8,28],[]]
	role_view.yaku = [[{"key":"three","points":5,"cards":[0,8,28]}],[]]
	for i in range(3):
		var card = table.take_card([0,8,28][i])
		card.position = Vector3(-2+i,0.012,0)
		card.rotation = table.FLAT
	table.sound_log.clear()
	await table.arrange_captures(role_view)
	check(table.sound_log[0].sound=="fuda-take", "Role movement did not start with fuda-take")
	check(table.sound_log.filter(func(e):return e.sound=="fuda-take").size()==1,"Role movement cue was repeated per card")
	check(table.sound_log.filter(func(e):return e.sound=="yaku").size()==3,"Formation lost per-card arrival sounds")
	await table.reset_deck()
	var capture = table.take_card(2)
	capture.position = Vector3(0,0.012,0)
	capture.rotation = table.FLAT
	role_view.captured = [[2],[]]
	role_view.yaku = [[],[]]
	table.sound_log.clear()
	await table.arrange_captures(role_view)
	check(table.sound_log.size()==1 and table.sound_log[0].sound=="fuda-turn", "Ordinary capture used the role-movement sound")
	print("GAME_MIGRATION_CHECK failures=%d overlaps=%d board=%d"%[errors.size(),audit.overlaps,audit.board_hits])
	quit(0 if errors.is_empty() else 1)
