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
	table.speed = 1
	# Real overlapping capture layers, rather than artificially separated cards.
	# Safety padding must not turn legal resting layers into endless delays.
	for i in range(16):
		var card = table.take_card(i)
		card.position = Vector3(-3+(i%8)*0.39,0.014+(i%8)*0.012,3.65 if i<8 else -3.65)
		card.rotation = table.FLAT
	var audit = load("res://tests/motion_audit.gd").new()
	var simultaneous := [0]
	var row_overlap := [false]
	var longest_gap := [0]
	var previous_tick := [Time.get_ticks_msec()]
	var observe := func():
		audit.inspect(table)
		longest_gap[0] = maxi(longest_gap[0],Time.get_ticks_msec()-previous_tick[0])
		previous_tick[0] = Time.get_ticks_msec()
		var moving: Array = table.bodies.filter(func(c):return c.flight and c.returning and c.elapsed>c.start_delay)
		simultaneous[0] = maxi(simultaneous[0],moving.size())
		row_overlap[0] = row_overlap[0] or (moving.any(func(c):return c.flight_from.z>0) and moving.any(func(c):return c.flight_from.z<0))
	physics_frame.connect(observe)
	var began := Time.get_ticks_msec()
	await table.reset_deck()
	var return_ms := Time.get_ticks_msec()-began
	physics_frame.disconnect(observe)
	check(return_ms<4000,"Sixteen captured cards took over four seconds at normal speed")
	check(longest_gap[0]<500,"Return planning stalled the physics loop for half a second")
	check(simultaneous[0]>=3 and row_overlap[0],"Horizontal returns serialized the cards or rows")
	check(audit.overlaps==0 and audit.board_hits==0 and table.stack.size()==48,"Return lost cards or crossed geometry")
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	table.sound_log.clear()
	began = Time.get_ticks_msec()
	await table.deal_cards(view,false)
	var deal_ms := Time.get_ticks_msec()-began
	check(deal_ms<6000,"Normal-speed 24-card deal took over six seconds")
	check(table.sound_log.filter(func(e):return e.sound in ["tefuda-place","bafuda-place"]).size()==24,"Fast dealing lost a card landing")
	# Upright paths stay close to the direct segment, without an outward loop.
	for from in [Vector3(-3,1,3.9),Vector3(3,3.55,12.2),Vector3(-17,-4,-12.2)]:
		var card = table.bodies[0]
		card.flight = false
		card.position = from
		card.move_to(table.STACK+Vector3(0,0.5,0),table.BACK_FLAT,1)
		card.route_to_stock()
		var direct: float = from.distance_to(card.flight_to)
		check(card.return_length<direct*1.6,"Quarter-arc return takes an unnecessary detour")
		check(absf(card.return_c2.x-card.flight_to.x)<0.001,"Upright return does not enter along the front/back stock axis")
		for i in range(101):
			var point: Vector3 = card.stock_point(i/100.0)
			var projection: float = (point-from).dot(card.flight_to-from)/direct
			check(projection>=-0.001 and projection<=direct+0.001,"Return goes backwards or past the stock")
		card.flight = false
	print("GAME_PACE_CHECK failures=%d return_ms=%d deal_ms=%d moving=%d planning_gap_ms=%d overlaps=%d board=%d"%[failures.size(),return_ms,deal_ms,simultaneous[0],longest_gap[0],audit.overlaps,audit.board_hits])
	quit(0 if failures.is_empty() else 1)
