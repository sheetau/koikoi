extends RefCounted
const GOLD = Color("e9ca85")
const INK = Color("241a22")
static func box(color: Color, border: Color = Color.TRANSPARENT, padding: int = 16) -> StyleBoxFlat:
	var style := StyleBoxFlat.new()
	style.bg_color = color
	style.border_color = border
	style.set_border_width_all(1)
	style.set_corner_radius_all(3)
	style.content_margin_left = padding
	style.content_margin_right = padding
	style.content_margin_top = 8
	style.content_margin_bottom = 8
	return style
static func theme(font: Font) -> Theme:
	var result := Theme.new()
	result.default_font = font
	result.default_font_size = 17
	for kind in ["Button","OptionButton","CheckButton"]:
		for state in ["normal","hover","pressed","hover_pressed","disabled","focus"]:
			result.set_stylebox(state,kind,box(Color("513c37") if state in ["hover","hover_pressed"] else Color(INK,0.86),GOLD))
		result.set_color("font_color",kind,Color("fff3d5"))
		result.set_color("font_hover_color",kind,GOLD)
		result.set_color("font_pressed_color",kind,GOLD)
	result.set_stylebox("panel","PopupMenu",box(Color(INK,0.97),GOLD))
	result.set_stylebox("hover","PopupMenu",box(Color("513c37"),GOLD))
	result.set_color("font_color","PopupMenu",Color("fff3d5"))
	result.set_color("font_hover_color","PopupMenu",GOLD)
	result.set_color("font_outline_color","PopupMenu",Color(0.2,0.7,0.8,0.35))
	result.set_constant("outline_size","PopupMenu",1)
	for kind in ["VScrollBar","HScrollBar"]:
		result.set_stylebox("scroll",kind,box(Color(GOLD,0.08),Color.TRANSPARENT,0))
		for state in ["grabber","grabber_highlight","grabber_pressed"]:
			var style := box(Color(GOLD,0.45 if state=="grabber" else 0.7),Color.TRANSPARENT,0)
			style.content_margin_top = 4
			style.content_margin_bottom = 4
			style.content_margin_left = 4
			style.content_margin_right = 4
			result.set_stylebox(state,kind,style)
	return result
static func pop(node: Node3D, delay: float = 0.0) -> void:
	var target := node.scale
	node.scale = target*0.001
	var tween := node.create_tween()
	if delay>0: tween.tween_interval(delay)
	tween.tween_property(node,"scale",target,0.24).set_trans(Tween.TRANS_BACK).set_ease(Tween.EASE_OUT)
static func dismiss(node: Node3D) -> void:
	if not is_instance_valid(node) or node.is_queued_for_deletion(): return
	var tween := node.create_tween()
	tween.tween_property(node,"scale",Vector3.ONE*0.001,0.16).set_trans(Tween.TRANS_QUAD).set_ease(Tween.EASE_IN)
	tween.tween_callback(node.queue_free)
static func reveal_list(world, label: Label) -> void:
	var surface: MeshInstance3D = world.surface
	var top: float = world.SIZE.y/2
	surface.scale.y = 0.001
	surface.position.y = top
	label.custom_minimum_size = label.get_minimum_size()
	label.visible_characters = 0
	var tween: Tween = world.create_tween().set_parallel()
	tween.tween_property(surface,"scale:y",1.0,0.28).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
	tween.tween_property(surface,"position:y",0.0,0.28).set_trans(Tween.TRANS_CUBIC).set_ease(Tween.EASE_OUT)
	tween.chain()
	var count := 0
	for line in label.text.split("\n"):
		count += line.length()+1
		tween.tween_property(label,"visible_characters",count,0.0)
		tween.tween_interval(0.08)
