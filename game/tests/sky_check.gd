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
	check(clouds.clouds.size()==12,"Cloud panorama lacks dispersed variants")
	var coverage_samples := 0
	for seconds in range(0,601,5):
		for x in range(0,4096,32):
			var count := 0
			for cloud in clouds.clouds:
				var extent: Vector2 = clouds.bounds(cloud,seconds)
				for wrap in [-4096,0,4096]:
					if x>=extent.x+wrap and x<=extent.y+wrap: count += 1
			check(count>0,"Panorama has an empty cloud longitude")
			check(count<=3,"More than three clouds overlap vertically")
			coverage_samples += 1
	var samples := 0
	for cloud in clouds.clouds:
		var min_widths: Array = cloud.shape.bands.map(func(b):return INF)
		var max_widths: Array = cloud.shape.bands.map(func(b):return 0.0)
		var displacement_range := 0.0
		var min_gaps: Array = cloud.clusters.map(func(c):return INF)
		var max_gaps: Array = cloud.clusters.map(func(c):return -INF)
		for seconds in range(0,601,2):
			for i in range(1,cloud.clusters.size()):
				var a: Dictionary = cloud.clusters[i-1]
				var b: Dictionary = cloud.clusters[i]
				var gap: float = clouds.anchor(cloud,b.low,seconds)-clouds.anchor(cloud,a.high,seconds)
				check(gap>70.0,"Independent cloud turns approached within their circular clearance")
				min_gaps[i] = minf(min_gaps[i],gap)
				max_gaps[i] = maxf(max_gaps[i],gap)
				displacement_range = maxf(displacement_range,max_gaps[i]-min_gaps[i])
				var previous: float = clouds.value(b.track,seconds-1)
				var current: float = clouds.value(b.track,seconds)
				var next: float = clouds.value(b.track,seconds+1)
				if (current-previous)*(next-current)<0:
					check(minf(absf(current-b.track.low),absf(current-b.track.high))<(b.track.high-b.track.low)*0.002,"Cloud span reverses before its clearance boundary")
			for i in range(cloud.shape.bands.size()):
				var band: Array = cloud.shape.bands[i]
				var interval: Vector2 = clouds.band_interval(cloud,i,seconds)
				var width := interval.y-interval.x
				var controls: Dictionary = cloud.bands[i]
				if controls.attached:
					var edge := interval.x if band[4]<0 else interval.y
					check(absf(edge-clouds.anchor(cloud,controls.parent,seconds))<0.01,"First inner U detached from its parent")
				check(width>=-0.001 and width<=band[1]-band[0]+0.001,"Swept U escaped its collision-safe corridor")
				min_widths[i] = minf(min_widths[i],width)
				max_widths[i] = maxf(max_widths[i],width)
				samples += 1
		check(displacement_range>150.0,"Cloud moves only as one rigid shape")
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
	# Compare actual sky pixels with/without the cloud mask at each bearing.
	# This also catches clouds that exist in the atlas but lie above the view.
	var empty_image := Image.create(1,1,false,Image.FORMAT_RGBA8)
	empty_image.fill(Color.TRANSPARENT)
	var empty_texture := ImageTexture.create_from_image(empty_image)
	var mask: Texture2D = table.sky_background.cloud_viewport.get_texture()
	var min_visible := 100000
	for seconds in [0.0,90.0]:
		clouds.clock = seconds
		clouds.queue_redraw()
		for bearing in range(12):
			table.camera.transform = saved
			table.camera.rotate_y(bearing*TAU/12)
			table.background_material.set_shader_parameter("cloud_mask",empty_texture)
			var plain := await frame()
			table.background_material.set_shader_parameter("cloud_mask",mask)
			var drawn := await frame()
			var visible_pixels := 0
			for y in range(0,drawn.get_height()/2,4):
				for x in range(0,drawn.get_width(),4):
					var a := plain.get_pixel(x,y)
					var b := drawn.get_pixel(x,y)
					if absf(a.r-b.r)+absf(a.g-b.g)+absf(a.b-b.b)>0.08: visible_pixels += 1
			min_visible = mini(min_visible,visible_pixels)
	check(min_visible>20,"Clouds disappeared from the normal camera at a panorama bearing")
	# The sun remains neutral white even inside the upper gold shadow.
	var latitude := 0.55*PI
	var longitude := 0.045*TAU
	var sun_direction := Vector3(sin(longitude)*sin(latitude),cos(latitude),-cos(longitude)*sin(latitude))
	table.camera.look_at(table.camera.position+sun_direction)
	var sun_frame := await frame()
	var sun_color := sun_frame.get_pixel(sun_frame.get_width()/2,sun_frame.get_height()/2)
	check(minf(sun_color.r,minf(sun_color.g,sun_color.b))>0.98,"Sun is tinted or darkened by the gold background")
	table.camera.transform = saved
	table.visible = true
	await table.screenshot("sky_final")
	print("GAME_SKY_CHECK failures=%d motion_samples=%d coverage_samples=%d min_visible_cloud_samples=%d world_ray_error=%.5f"%[failures.size(),samples,coverage_samples,min_visible,error])
	quit(0 if failures.is_empty() else 1)
