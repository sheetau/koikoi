extends Node2D
## User-drawn cloud centerlines: circular turns, swept U bands and rigid Y.
const WIDTH := 4096.0
const STROKE := 10.0
var shapes: Array = JSON.parse_string(FileAccess.get_file_as_string("res://assets/sky/cloud_shapes.json"))
var clouds: Array = []
var clock := 0.0
var accumulator := 0.0
var rng := RandomNumberGenerator.new()
func _ready() -> void:
	rng.seed = 60239 # Presentation randomness is independent of game state.
	for index in range(6):
		var shape: Dictionary = shapes[index%shapes.size()].duplicate(true)
		var anchors: Array[float] = []
		for arc in shape.arcs: anchors.append(arc[0])
		for point in shape.ends: anchors.append(point[0])
		anchors.sort()
		var clusters: Array = []
		for x in anchors:
			if clusters.is_empty() or x-clusters[-1].high>160:
				clusters.append({"low":x,"high":x})
			else: clusters[-1].high = x
		for i in range(clusters.size()):
			var group: Dictionary = clusters[i]
			var margin := 140.0
			if i>0: margin = minf(margin,(group.low-clusters[i-1].high)*0.20)
			if i+1<clusters.size(): margin = minf(margin,(clusters[i+1].low-group.high)*0.20)
			group.track = track(margin)
		var bands: Array = []
		for band in shape.bands:
			# Original bands sit on the inside of their parent turn. Keep their
			# entire swept interval in that corridor, including the stroke.
			bands.append({"drift":track(1.0),"width":track(1.0)})
		var scale_factor: float = rng.randf_range(1050,1400)/shape.size[0]
		clouds.append({"shape":shape,"clusters":clusters,"bands":bands,"scale":scale_factor,
			"origin":Vector2((index%2)*2048-600+rng.randf_range(-100,100),100+(index/2)*145),"drift":track(70.0)})
	queue_redraw()
func track(amplitude: float) -> Dictionary:
	return {"amplitude":amplitude,"f1":rng.randf_range(0.07,0.16),"f2":rng.randf_range(0.17,0.29),"p1":rng.randf()*TAU,"p2":rng.randf()*TAU}
func value(t: Dictionary, seconds: float) -> float:
	return t.amplitude*(0.65*sin(seconds*t.f1+t.p1)+0.35*sin(seconds*t.f2+t.p2))
func anchor(cloud: Dictionary, x: float, seconds: float) -> float:
	var nearest: Dictionary = cloud.clusters[0]
	var distance := INF
	for group in cloud.clusters:
		var gap := absf(x-clampf(x,group.low,group.high))
		if gap<distance:
			distance = gap
			nearest = group
	return x+value(nearest.track,seconds)
func band_interval(cloud: Dictionary, index: int, seconds: float) -> Vector2:
	var band: Array = cloud.shape.bands[index]
	var controls: Dictionary = cloud.bands[index]
	var corridor: float = band[1]-band[0]
	var phase: float = seconds*controls.width.f1+controls.width.p1+0.45*sin(seconds*controls.width.f2+controls.width.p2)
	var width := corridor*(0.5+0.5*sin(phase))
	var offset := (corridor-width)*(0.5+0.5*value(controls.drift,seconds))
	var parent: float = band[0] if band[4]<0 else band[1]
	var delta := anchor(cloud,parent,seconds)-parent
	return Vector2(band[0]+offset+delta,band[0]+offset+width+delta)
func semicircle(x: float, top: float, bottom: float, side: float) -> PackedVector2Array:
	var points := PackedVector2Array()
	var radius := (bottom-top)/2
	for i in range(33):
		var angle := PI*i/32.0
		points.append(Vector2(x+side*radius*sin(angle),(top+bottom)/2-radius*cos(angle)))
	return points
func _draw() -> void:
	for cloud in clouds:
		var shape: Dictionary = cloud.shape
		for wrap in [-WIDTH,0.0,WIDTH]:
			var origin: Vector2 = cloud.origin+Vector2(wrap+value(cloud.drift,clock),0)
			if origin.x>WIDTH+200 or origin.x+shape.size[0]*cloud.scale< -200: continue
			draw_set_transform(origin,0,Vector2.ONE*cloud.scale)
			for line in shape.lines:
				draw_line(Vector2(anchor(cloud,line[0],clock),line[2]),Vector2(anchor(cloud,line[1],clock),line[2]),Color.WHITE,STROKE,true)
			for arc in shape.arcs:
				draw_polyline(semicircle(anchor(cloud,arc[0],clock),minf(arc[1],arc[2]),maxf(arc[1],arc[2]),arc[3]),Color.WHITE,STROKE,true)
			for point in shape.ends:
				draw_circle(Vector2(anchor(cloud,point[0],clock),point[1]),STROKE/2,Color.WHITE,true,-1,true)
			for i in range(shape.bands.size()):
				var band: Array = shape.bands[i]
				var interval := band_interval(cloud,i,clock)
				var a := semicircle(interval.x,band[2],band[3],band[4])
				var b := semicircle(interval.y,band[2],band[3],band[4])
				# A convex strip remains valid even as the U collapses to one
				# stroke; a single concave polygon becomes numerically singular.
				if interval.y-interval.x>0.05:
					for j in range(a.size()-1):
						draw_primitive(PackedVector2Array([a[j],a[j+1],b[j+1],b[j]]),PackedColorArray([Color.WHITE]),PackedVector2Array())
				b.reverse()
				a.append_array(b)
				a.append(a[0])
				draw_polyline(a,Color.WHITE,STROKE,true)
	draw_set_transform(Vector2.ZERO)
func _process(delta: float) -> void:
	clock += delta
	accumulator += delta
	if accumulator>=1.0/30.0:
		accumulator = 0
		queue_redraw()
