extends SceneTree
var failures: Array[String] = []
func _initialize() -> void:
	call_deferred("run")
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.speed = 4
	for i in range(2):
		var card = table.take_card(i*4)
		card.position = Vector3(-0.7+i*1.4,2.25,0)
		card.rotation = Vector3(0,PI*(1-i),0)
	var audit = load("res://tests/motion_audit.gd").new()
	var simultaneous := [0]
	var observe := func():
		audit.inspect(table)
		simultaneous[0] = maxi(simultaneous[0],table.bodies.filter(func(c):return c.flight and c.returning).size())
	physics_frame.connect(observe)
	await table.reset_deck(true)
	physics_frame.disconnect(observe)
	if simultaneous[0]!=1 or audit.overlaps!=0 or audit.board_hits!=0 or table.stack.size()!=48:
		failures.append("Dealer return was not sequential, collision-free and complete")
	for message in failures:
		push_error(message)
	print("GAME_DEALER_RETURN_CHECK failures=%d simultaneous=%d overlaps=%d board=%d"%[failures.size(),simultaneous[0],audit.overlaps,audit.board_hits])
	quit(0 if failures.is_empty() else 1)
