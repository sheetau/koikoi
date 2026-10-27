extends SceneTree
## Synthetic public role membership and settings interactions.
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
	table.speed = 8
	table.qa = true
	var view: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json")).view
	view.hand = [-1,-1,-1,-1,-1,-1,-1,-1]
	view.opponent_slots = []
	view.field = []
	view.captured = [[0,8,28,32,1,5,9],[20,24,36,4,12]]
	view.yaku = [[{"key":"three","points":5,"cards":[0,8,28]},{"key":"flower","points":1,"cards":[8,32]},{"key":"moon","points":1,"cards":[28,32]},{"key":"red","points":5,"cards":[1,5,9]}],[{"key":"animals","points":5,"cards":[20,24,36]},{"key":"tane","points":1,"cards":[4,12,20,24,36]}]]
	table.state = view
	table.started = true
	await table.restore_view(view)
	table.clear_menu()
	# Face the human role rack to inspect all combined names from the front.
	table.camera.position = Vector3(3,7,-8)
	table.camera.look_at(Vector3(3,2,12))
	var groups: Array = table.presentation_roles(view,0)
	check(groups.size()==2 and groups[0].roles.size()==3 and groups[0].cards.size()==4,"Transitive overlaps did not merge into one unique-card row")
	check(table.label_nodes.size()==4,"Overlapping role labels created separate rows")
	var plaque = table.label_nodes[0]
	check(plaque.get_child(1).text.contains("三光") and plaque.get_child(1).text.contains("花見酒") and plaque.get_child(1).text.contains("月見酒"),"Shared row lost a role name")
	for id in [0,8,28,32]:
		check(is_equal_approx(table.known[id].position.y,plaque.position.y),"Shared member is not on its label row")
	var fixed: Dictionary = {}
	for id in view.captured[0]:
		fixed[id] = table.known[id].transform
	await table.screenshot("grouped_roles_ja")
	await table.switch_language()
	check(plaque.get_child(1).text.contains("FLOWER") and plaque.get_child(1).text.contains("MOON"),"Language switch lost grouped titles")
	await table.screenshot("grouped_roles_en")
	table.sound_log.clear()
	await table.sweep_yaku(view,0)
	for id in fixed:
		check(table.known[id].transform.is_equal_approx(fixed[id]),"Grouped total-yaku moved the winner")
	for id in [20,24,36]:
		check(is_equal_approx(table.known[id].position.y,0.51),"Transferred shared roles did not append after two winner rows")
	check(table.sound_log.filter(func(e):return e.sound=="fuda-take").size()==2,"Separate count role lost the take cue")
	check(table.sound_log.filter(func(e):return e.sound=="yaku").is_empty(),"Transfer replayed formation audio")
	await table.reset_deck()
	check(table.stack.size()==48,"Shared return lost cards")
	# The saved setting affects the next match; the live HUD uses server metadata.
	view.rounds = 6
	table.state = view
	table.refresh_hud()
	check(table.left_label.text.contains("/ 06"),"HUD ignored current match length")
	await table.open_overlay("settings")
	var options = table.overlay_box.find_child("MatchRounds",true,false)
	check(options.item_count==4,"Round setting is missing options")
	options.select(0)
	options.item_selected.emit(0)
	check(table.match_rounds==1 and table.state.rounds==6,"Setting changed the running match")
	# Restore the ordinary camera for settings screenshots.
	table.update_camera()
	table.busy = true
	await table.open_overlay("settings")
	check(table.overlay_box.find_child("RestartMatch",true,false).disabled,"Restart can interrupt in-flight events")
	table.busy = false
	await table.open_overlay("restart")
	check(table.overlay_box.get_child_count()==3,"Restart confirmation is missing")
	await table.screenshot("restart_en")
	print("GAME_GROUPED_ROLES_CHECK failures=%d"%failures.size())
	quit(0 if failures.is_empty() else 1)
