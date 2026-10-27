extends SceneTree
func _initialize() -> void:
	call_deferred("run")
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	table.speed = 3
	var audit = load("res://tests/motion_audit.gd").new()
	physics_frame.connect(func(): audit.inspect(table))
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	view.hand = [-1,-1,-1,-1,-1,-1,-1,-1]
	view.opponent_slots = []
	view.field = [18,19,20,21,24,25,28,29]
	view.captured = [[2,3,6,7,10,11,14,15,4,12,16,1,5,9,0], []]
	view.yaku = [[], []]
	view.pending = -1
	table.state = view
	await table.restore_view(view)
	for ids in [[18],[20],[21],[28]]:
		for id in ids:
			view.field.erase(id)
			view.captured[0].append(id)
		table.slide_field(view.field)
		await table.arrange_captures(view)
	await table.reset_deck()
	print("GAME_CAPTURE_MOTION overlaps=%d board=%d lift=%.3f" % [audit.overlaps, audit.board_hits, audit.max_lift])
	quit(0 if audit.overlaps == 0 and audit.board_hits == 0 else 1)
