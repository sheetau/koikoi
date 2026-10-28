extends Node2D
## Circular cloud turns joined by independently extensible horizontal spans.
const WIDTH := 4096.0
const STROKE := 10.0
const CLOUD_SCALE := 0.25
const SLOT := WIDTH/4
var shapes: Array = JSON.parse_string(FileAccess.get_file_as_string("res://assets/sky/cloud_shapes.json"))
var clouds: Array = []
var clock := 0.0
var accumulator := 0.0
var rng := RandomNumberGenerator.new()
func _ready() -> void:
	rng.seed = 60239
	# Four separated slots in each of three lanes. Staggering the lanes
	# covers the full panorama without ever stacking more than three clouds.
	for index in range(12):
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
		var fixed := 0.0
		var minimum := 0.0
		var weights: Array[float] = []
		for i in range(clusters.size()):
			fixed += clusters[i].high-clusters[i].low
			if i>0:
				var gap: float = clusters[i].low-clusters[i-1].high
				clusters[i].minimum = maxf(130,gap*0.9)
				minimum += clusters[i].minimum
				weights.append(gap*rng.randf_range(0.7,1.3))
		var extra := (SLOT-100)/CLOUD_SCALE-70-fixed-minimum
		var weight_sum: float = weights.reduce(func(a,b):return a+b,0.0)
		for i in range(1,clusters.size()):
			clusters[i].track = track(clusters[i].minimum,clusters[i].minimum+extra*weights[i-1]/weight_sum)
		var bands: Array = []
		for band in shape.bands:
			var parent: float = band[0] if band[4]<0 else band[1]
			var reference: float = parent
			var distance := INF
			for arc in shape.arcs:
				if arc[3]!=band[4] or absf(minf(arc[1],arc[2])-band[2])>0.05: continue
				var d: float = absf(arc[0]-reference)
				if d<distance:
					distance = d
					parent = arc[0]
			bands.append({"parent":parent,"attached":distance<0.05,"drift":track(0,1),"width":track(0,1)})
		clouds.append({"shape":shape,"clusters":clusters,"bands":bands,"scale":CLOUD_SCALE,
			"origin":Vector2((index%4)*SLOT+(index/4)*SLOT/3,220+(index/4)*40),
			"mirror":-1.0 if index%2 else 1.0,"drift":track(-12,12,100,170)})
	queue_redraw()
func track(low: float, high: float, shortest: float = 40, longest: float = 105) -> Dictionary:
	# A track keeps its direction until reaching a corridor boundary. Each
	# traversal lasts tens of seconds and eases to zero speed at the boundary.
	return {"low":low,"high":high,"duration":rng.randf_range(shortest,longest),"phase":rng.randf_range(0,2)}
func value(t: Dictionary, seconds: float) -> float:
	var phase := fposmod(seconds/t.duration+t.phase,2.0)
	var progress := phase if phase<1.0 else 2.0-phase
	return lerpf(t.low,t.high,0.5-0.5*cos(PI*progress))
func offsets(cloud: Dictionary, seconds: float) -> PackedFloat32Array:
	if cloud.get("offset_time",-1.0)==seconds: return cloud.offsets
	var result := PackedFloat32Array([0.0])
	for i in range(1,cloud.clusters.size()):
		var before: Dictionary = cloud.clusters[i-1]
		result.append(result[-1]+before.high-before.low+value(cloud.clusters[i].track,seconds))
	var total: float = result[-1]+cloud.clusters[-1].high-cloud.clusters[-1].low
	for i in range(result.size()): result[i] -= total/2
	cloud.offset_time = seconds
	cloud.offsets = result
	return result
func anchor(cloud: Dictionary, x: float, seconds: float) -> float:
	var selected := 0
	var distance := INF
	for i in range(cloud.clusters.size()):
		var group: Dictionary = cloud.clusters[i]
		var gap := absf(x-clampf(x,group.low,group.high))
		if gap<distance:
			distance = gap
			selected = i
	return offsets(cloud,seconds)[selected]+x-cloud.clusters[selected].low
func bounds(cloud: Dictionary, seconds: float) -> Vector2:
	var positions := offsets(cloud,seconds)
	var radius := 38.0 # Covers every circular turn plus the round stroke cap.
	var half_width: float = (-positions[0]+radius)*cloud.scale
	var center: float = cloud.origin.x+value(cloud.drift,seconds)
	return Vector2(center-half_width,center+half_width)
func band_interval(cloud: Dictionary, index: int, seconds: float) -> Vector2:
	var band: Array = cloud.shape.bands[index]
	var controls: Dictionary = cloud.bands[index]
	var width: float = (band[1]-band[0])*value(controls.width,seconds)
	var parent: float = anchor(cloud,controls.parent,seconds)
	if controls.attached:
		return Vector2(parent,parent+width) if band[4]<0 else Vector2(parent-width,parent)
	var offset: float = (band[1]-band[0]-width)*value(controls.drift,seconds)
	return Vector2(band[0]+offset,band[0]+offset+width)+Vector2.ONE*(parent-controls.parent)
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
			if origin.x-SLOT/2>WIDTH or origin.x+SLOT/2<0: continue
			draw_set_transform(origin,0,Vector2(cloud.mirror,1)*cloud.scale)
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
