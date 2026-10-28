extends SceneTree
## Public synthetic cards isolate U-turns and dense simultaneous returns.
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
	var audit = load("res://tests/motion_audit.gd").new()
	physics_frame.connect(func(): audit.inspect(table))
	table.speed = 1.0
	for player in range(2):
		var cards: Array = []
		for i in range(3):
			var card = table.take_card(2 + i * 4)
			card.position = table.Layout.capture(card.identity, i, player)
			card.rotation = table.FLAT if player == 0 else Vector3(-PI / 2, 0, PI)
			cards.append(card)
			var sign_x := 1 if player == 0 else -1
			table.fly(card, Vector3((3.0 - i * 0.88) * sign_x, 3.55, 12.2 * sign_x), Vector3(0, PI if player == 0 else 0, 0), 0.85 + i * 0.055)
			card.route_to_role(player)
			var last_x: float = card.flight_from.x
			var reversed := false
			for sample in range(1, 101):
				var pose: Transform3D = card.pose_at(card.duration * sample / 100.0)
				var dx: float = (pose.origin.x - last_x) * card.role_side
				if dx < -0.0001:
					reversed = true
					check(absf(pose.basis.get_rotation_quaternion().dot(card.rotation_to)) > 0.9999, "Role alignment continued after the U-turn")
				last_x = pose.origin.x
			check(reversed, "Role route did not turn back")
		table.stagger_roles(cards)
		table.plan_flights()
		var arrived: Array = []
		while cards.any(func(c): return c.flight):
			await physics_frame
			for card in cards:
				if not card.flight and card.identity not in arrived:
					arrived.append(card.identity)
		check(arrived == [2, 6, 10], "Role cards did not arrive from the left")
		await table.reset_deck()
	# A long chaff row can extend beyond the nominal board-side turn.
	var settled = table.take_card(46)
	settled.position = Vector3(5.8, 3.55, -12.2)
	var entering = table.take_card(30)
	entering.position = Vector3(-3.31, 0.031, -0.68)
	entering.rotation = table.FLAT
	table.fly(entering, Vector3(6.68, 3.55, -12.2), Vector3.ZERO)
	entering.route_to_role(1)
	check(entering.role_edge > entering.flight_to.x + 0.8, "Long row's turn is inside its last card")
	table.stagger_roles([entering])
	table.plan_flights()
	for sample in range(101):
		check(not audit.intersects(entering.pose_at(entering.duration * sample / 100.0), settled.transform), "Long row arrival crossed an earlier card")
	await table.finish_flights()
	await table.reset_deck()
	# A readable role-to-role wait must not coarsen collision sampling in the
	# short moving interval. This pickup crosses a resting card at its height.
	for wait in [0.0,0.65,1.3]:
		table.speed = 12
		var obstacle = table.take_card(40)
		obstacle.position = Vector3(3.69,0.014,-3.65)
		obstacle.rotation = table.FLAT
		var delayed = table.take_card(33)
		delayed.position = Vector3(1.54,0.014,-3.65)
		delayed.rotation = table.FLAT
		table.fly(delayed,Vector3(-0.36,-1.01,-12.2),Vector3.ZERO)
		delayed.route_to_role(1)
		delayed.role_rebound = true
		delayed.duration = wait+0.24
		delayed.start_delay = wait
		table.plan_flights()
		var crossed := false
		for sample in range(501):
			crossed = crossed or audit.intersects(delayed.pose_at(wait+sample*0.24/500.0),obstacle.transform)
		check(not crossed,"Delayed role crossed a resting capture card")
		await table.finish_flights()
		await table.reset_deck()
	# Different role rows share the outer turn but must remain separated.
	table.speed = 12
	for item in [[21, 1.54, -0.36, 2.03, 0.15402866091047], [36, -0.61, -1.24, 3.55, 0.1584929443541]]:
		var card = table.take_card(int(item[0]))
		card.position = Vector3(item[1], 0.014, -5.65)
		card.rotation = Vector3(-PI / 2, 0, PI)
		table.fly(card, Vector3(item[2], item[3], -12.2), Vector3.ZERO)
		card.route_to_role(1, 6.5)
		card.duration = item[4]
		card.start_delay = card.duration - (card.role_lengths.x + card.role_lengths.y + card.role_lengths.z) / 14.0 / table.speed
	table.plan_flights()
	for step in range(1, 170):
		check(not audit.intersects(table.known[21].pose_at(step / 1000.0), table.known[36].pose_at(step / 1000.0)), "Different yaku rows intersect during the outer turn")
	await table.finish_flights()
	await table.reset_deck()
	table.speed = 1
	# Dense upright rows also fold and return without crossing one another.
	for i in range(48):
		var card = table.take_card(i)
		card.position = Vector3((i % 8 - 3.5) * 0.88, 3.55 - (i / 8) * 1.52, 12.2)
		card.rotation = Vector3(0, PI, 0)
	await table.reset_deck()
	check(table.stack.size() == 48, "Dense return lost a card")
	# A large chaff role remains one long row, including during settlement.
	for i in range(24):
		var card = table.take_card(i)
		card.position = Vector3(3.0 - i * 0.88, 3.55, 12.2)
		card.rotation = Vector3(0, PI, 0)
	await table.reset_deck()
	check(table.stack.size() == 48, "Long-row return lost a card")
	check(audit.overlaps == 0 and audit.board_hits == 0, "Role/return motion intersects geometry")
	print("GAME_ROLE_MOTION failures=%d overlaps=%d board=%d" % [errors.size(), audit.overlaps, audit.board_hits])
	quit(0 if errors.is_empty() else 1)
