extends SceneTree
## Public presentation fixtures for the five imported dashboard tasks.
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
	table.speed = 5
	var fixture: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json"))
	var maximum := [0]
	var observe_deal := func(): maximum[0] = maxi(maximum[0],table.bodies.filter(func(c): return c.flight and c.arrival_sound in ["tefuda-place","bafuda-place"]).size())
	physics_frame.connect(observe_deal)
	await table.opening(fixture.events[0])
	physics_frame.disconnect(observe_deal)
	check(maximum[0] == 1, "Dealer and hand/field dealing must move one card at a time")
	var lands: Array = table.sound_log.filter(func(e): return e.sound in ["tefuda-place","bafuda-place"])
	check(lands.size() == 26, "Single-card dealing lost an arrival")
	for batch in range(4):
		for i in range(6):
			check(lands[2+batch*6+i].sound == ("bafuda-place" if i<2 else "tefuda-place"), "Single dealing changed batch order")
	await table.reset_deck()
	# Existing winner row/slot order can differ from the new scoring list order.
	for winner in range(2):
		var view: Dictionary = fixture.view.duplicate(true)
		view.hand = [-1,-1,-1,-1,-1,-1,-1,-1]
		view.opponent_slots = []
		view.field = []
		view.captured = [[0,8,28,1,5,9],[20,24,36,2,3,6,7,10,11,14,15,18,19]]
		view.yaku = [[{"key":"three","points":5,"cards":[0,8,28]},{"key":"red","points":5,"cards":[1,5,9]}],[{"key":"animals","points":5,"cards":[20,24,36]},{"key":"kasu","points":1,"cards":[2,3,6,7,10,11,14,15,18,19]}]]
		if winner==1:
			view.captured.reverse()
			view.yaku.reverse()
		table.ending = {}
		await table.restore_view(view)
		var settled: Dictionary = {}
		for id in view.captured[winner]:
			settled[id] = table.known[id].transform
		view.yaku[winner].reverse()
		for row in view.yaku[winner]:
			row.cards.reverse()
		var stayed := [true]
		var observe := func():
			for id in settled:
				stayed[0] = stayed[0] and table.known[id].transform.is_equal_approx(settled[id]) and not table.known[id].flight
		physics_frame.connect(observe)
		table.sound_log.clear()
		await table.sweep_yaku(view,winner)
		physics_frame.disconnect(observe)
		check(stayed[0], "Total-yaku transfer moved or swapped winner cards")
		for row in view.yaku[1-winner]:
			var heights: Array = row.cards.map(func(id): return table.known[id].position.y)
			check(heights.max()-heights.min()<0.001 and heights.max()<2.03, "Opponent yaku did not append below winner rows")
		check(table.sound_log.filter(func(e): return e.sound=="fuda-take").size()==2, "Transfer needs exactly one take cue per role")
		check(table.sound_log.filter(func(e): return e.sound=="yaku").is_empty(), "Transfer replayed formation sounds")
		await table.reset_deck()
	# Capturing the sake cup makes both viewing roles: notify them separately.
	var old: Dictionary = fixture.view.duplicate(true)
	old.hand = [-1,-1,-1,-1,-1,-1,-1,-1]
	old.opponent_slots = []
	old.field = [33]
	old.captured = [[8,28],[]]
	old.yaku = [[],[]]
	old.player = 0
	old.phase = 0
	old.legal = []
	table.state = old
	await table.restore_view(old)
	var played = table.take_card(32)
	played.position = Vector3(0,1.04,3.9)
	played.rotation = Vector3.ZERO
	var next: Dictionary = old.duplicate(true)
	next.field = []
	next.captured = [[8,28,32,33],[]]
	next.yaku = [[{"key":"flower","points":1,"cards":[8,32]},{"key":"moon","points":1,"cards":[28,32]}],[]]
	next.phase = 3
	next.legal = [48,49]
	table.clear_menu()
	table.announcement_history.clear()
	var timing: Array = []
	var seen: Array = []
	var observe_notice := func():
		if not table.announcement.text.is_empty() and table.announcement.text not in seen:
			seen.append(table.announcement.text)
			timing.append(Time.get_ticks_msec())
			check(table.menu_buttons.is_empty(), "Koi buttons appeared during role notification")
	physics_frame.connect(observe_notice)
	await table.animate_event({"kind":"capture","card":32,"targets":[33],"player":0,"new_yaku":true,"view":next})
	physics_frame.disconnect(observe_notice)
	check(seen==["花見酒です。","月見酒です。"], "New roles were not announced separately in order")
	check(timing.size()==2 and timing[1]-timing[0]>=980, "Role notice disappeared too soon")
	table.started = true
	table.show_decision()
	var labels: Array = table.menu.get_children().filter(func(c): return c is Label3D).map(func(c): return c.text)
	check(labels.size()==3 and table.word("koi_question") in labels and "はい" in labels and "いいえ" in labels, "Koi prompt must contain only the question and choices")
	await table.reset_deck()
	# Ordinary play duration is less than half the previous 0.62 second flight.
	var discard = table.take_card(2)
	discard.position = Vector3(0,1.04,3.9)
	var discarded: Dictionary = old.duplicate(true)
	discarded.field = [2]
	await table.animate_event({"kind":"place","card":2,"targets":[],"player":0,"new_yaku":false,"view":discarded})
	check(discard.duration*table.speed<=0.301, "Field play was not accelerated")
	print("GAME_BACKLOG_CHECK failures=%d single_deal_max=%d notices=%s" % [failures.size(),maximum[0],seen])
	quit(0 if failures.is_empty() else 1)
