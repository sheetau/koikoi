extends SceneTree
var failures: Array[String] = []
func _initialize() -> void: call_deferred("run")
func check(ok: bool, message: String) -> void:
	if not ok and message not in failures:
		failures.append(message)
		push_error(message)
func frame() -> Image:
	await process_frame
	await process_frame
	RenderingServer.force_draw(false)
	return root.get_texture().get_image()
func run() -> void:
	var table = load("res://main.tscn").instantiate()
	table.testing = true
	root.add_child(table)
	await create_timer(0.8).timeout
	table.qa = true
	var clouds = table.sky_background.clouds
	clouds.set_process(false)
	table.background_material.set_shader_parameter("animation_time",0.0)
	check(clouds.shapes.size()==5,"User cloud variants were lost")
	check(clouds.clouds.all(func(c):return is_equal_approx(c.scale,0.25)),"Cloud variants have unequal or oversized global scales")
	var samples := 0
	for cloud in clouds.clouds:
		var min_widths: Array = cloud.shape.bands.map(func(b):return INF)
		var max_widths: Array = cloud.shape.bands.map(func(b):return 0.0)
		var displacement_range := 0.0
		for seconds in range(0,601,2):
			for i in range(1,cloud.clusters.size()):
				var a: Dictionary = cloud.clusters[i-1]
				var b: Dictionary = cloud.clusters[i]
				var gap: float = clouds.anchor(cloud,b.low,seconds)-clouds.anchor(cloud,a.high,seconds)
				check(gap>70.0,"Independent cloud turns approached within their circular clearance")
				var da: float = clouds.anchor(cloud,a.low,seconds)-a.low
				var db: float = clouds.anchor(cloud,b.low,seconds)-b.low
				displacement_range = maxf(displacement_range,absf(da-db))
			for i in range(cloud.shape.bands.size()):
				var band: Array = cloud.shape.bands[i]
				var interval: Vector2 = clouds.band_interval(cloud,i,seconds)
				var width := interval.y-interval.x
				check(width>=-0.001 and width<=band[1]-band[0]+0.001,"Swept U escaped its collision-safe corridor")
				min_widths[i] = minf(min_widths[i],width)
				max_widths[i] = maxf(max_widths[i],width)
				samples += 1
		check(displacement_range>10.0,"Cloud moves only as one rigid shape")
		for i in range(min_widths.size()):
			var width: float = cloud.shape.bands[i][1]-cloud.shape.bands[i][0]
			check(min_widths[i]<width*0.12 and max_widths[i]>width*0.85,"Double U does not broaden and collapse")
	# Actual rendered frames: vertical extent remains constant while partial
	# horizontal spans and double-U bands visibly vary over time.
	var atlas_bounds: Array = []
	for seconds in [0.0,35.0,90.0]:
		clouds.clock = seconds
		clouds.queue_redraw()
		await frame()
		var atlas: Image = table.sky_background.cloud_viewport.get_texture().get_image()
		atlas_bounds.append(atlas.get_used_rect())
		check(atlas.get_used_rect().end.y<atlas.get_height(),"Cloud atlas clips a lower turn")
		await table.screenshot("sky_clouds_%d"%seconds)
	for bounds in atlas_bounds:
		check(bounds.position.y==atlas_bounds[0].position.y and bounds.size.y==atlas_bounds[0].size.y,"Cloud animation stretched or shifted vertically")
	# Sample the same world rays from two camera angles. A view-following
	# vignette would change them; fixed background latitudes must not.
	table.visible = false
	var saved: Transform3D = table.camera.transform
	var rays: Array[Vector3] = []
	for x in [-0.15,0.0,0.15]:
		for y in [-0.9,-1.0,-1.1]: rays.append(Vector3(x,y,-1).normalized())
	var baseline: Array[Color] = []
	var error := 0.0
	for turn in range(2):
		table.camera.look_at(table.camera.position+Vector3(0,-1.0-turn*0.2,-1))
		var rendered := await frame()
		for i in range(rays.size()):
			var point: Vector2i = Vector2i(table.camera.unproject_position(table.camera.position+rays[i]*100))
			var color := rendered.get_pixelv(point)
			if turn==0: baseline.append(color)
			else: error += absf(color.r-baseline[i].r)+absf(color.g-baseline[i].g)+absf(color.b-baseline[i].b)
	error /= rays.size()*3
	check(error<0.04,"Background shade follows the camera")
	table.camera.transform = saved
	table.visible = true
	await table.screenshot("sky_final")
	print("GAME_SKY_CHECK failures=%d motion_samples=%d world_ray_error=%.5f"%[failures.size(),samples,error])
	quit(0 if failures.is_empty() else 1)
