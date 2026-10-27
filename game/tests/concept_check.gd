extends SceneTree
func _initialize() -> void:
    call_deferred("run")
func run() -> void:
    var table = load("res://main.tscn").instantiate()
    table.testing = true
    root.add_child(table)
    await create_timer(0.2).timeout
    table.qa = true
    table.speed = 3.0
    var fixture = JSON.parse_string(FileAccess.get_file_as_string("res://tests/opening_view.json"))
    table.state = fixture.view
    table.started = true
    table.clear_menu()
    var audit = load("res://tests/motion_audit.gd").new()
    physics_frame.connect(func(): audit.inspect(table))
    await table.deal_cards(fixture.view)
    assert(table.stack.size() == 24)
    assert(table.bodies.filter(func(c): return c.visible).size() == 24)
    table.refresh_hud()
    await table.screenshot("concept_table")
    table.state.phase = 3
    table.state.player = 0
    table.state.yaku_points[0] = 5
    table.show_decision()
    await table.screenshot("concept_question")
    table.open_overlay("settings")
    await process_frame
    await table.screenshot("concept_settings")
    await table.reset_deck()
    assert(table.stack.size() == 48)
    assert(table.bodies.all(func(c): return not c.visible))
    print("GAME_CONCEPT_CHECK max_lift=%.3f overlaps=%d board=%d" % [audit.max_lift, audit.overlaps, audit.board_hits])
    quit(0 if audit.overlaps == 0 and audit.board_hits == 0 else 1)
