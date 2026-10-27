extends SceneTree
func _initialize() -> void:
	call_deferred("run")
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await process_frame
	var base = table.take_card(0)
	base.position = Vector3(0, 0.012, 0)
	base.rotation = table.FLAT
	var incoming = table.take_card(8)
	incoming.position = Vector3(0.13, 0.512, 0.13)
	incoming.rotation = table.FLAT
	await table.impact_capture(incoming, [0])
	print("GAME_IMPACT peak=%.3f top=%s base=%s" % [table.last_burst_peak, incoming.position, base.position])
	quit(0 if table.last_burst_peak > 0.25 and table.last_impact_settled else 1)
