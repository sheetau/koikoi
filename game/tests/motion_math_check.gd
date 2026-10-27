extends SceneTree
func _initialize() -> void:
    var audit = load("res://tests/motion_audit.gd").new()
    var rng := RandomNumberGenerator.new()
    rng.seed = 71203
    var pairs: Array = []
    for i in range(12000):
        var a := Transform3D(Basis.from_euler(Vector3(rng.randf_range(-PI,PI),rng.randf_range(-PI,PI),rng.randf_range(-PI,PI))),Vector3.ZERO)
        var scale := 0.6 if i % 3 == 0 else 8.0
        var b := Transform3D(Basis.from_euler(Vector3(rng.randf_range(-PI,PI),rng.randf_range(-PI,PI),rng.randf_range(-PI,PI))),Vector3(rng.randf_range(-scale,scale),rng.randf_range(-scale,scale),rng.randf_range(-scale,scale)))
        pairs.append([a,b])
    var reference: Array = []
    var start := Time.get_ticks_usec()
    for pair in pairs:
        reference.append(reference_intersects(pair[0],pair[1]))
    var baseline := Time.get_ticks_usec()-start
    var failures := 0
    start = Time.get_ticks_usec()
    for i in range(pairs.size()):
        if audit.intersects(pairs[i][0],pairs[i][1]) != reference[i]:
            failures += 1
    var optimized := Time.get_ticks_usec()-start
    print("GAME_MOTION_MATH pairs=%d mismatches=%d reference_ms=%.2f optimized_ms=%.2f speedup=%.2f" % [pairs.size(),failures,baseline/1000.0,optimized/1000.0,float(baseline)/optimized])
    quit(0 if failures == 0 else 1)
func reference_intersects(a: Transform3D, b: Transform3D) -> bool:
    var axes: Array[Vector3] = [a.basis.x,a.basis.y,a.basis.z,b.basis.x,b.basis.y,b.basis.z]
    for x in [a.basis.x,a.basis.y,a.basis.z]:
        for y in [b.basis.x,b.basis.y,b.basis.z]:
            if x.cross(y).length_squared() > 0.00001:
                axes.append(x.cross(y).normalized())
    var half := Vector3(0.39,0.64,0.001)
    for axis in axes:
        var ar: float = absf(axis.dot(a.basis.x))*half.x + absf(axis.dot(a.basis.y))*half.y + absf(axis.dot(a.basis.z))*half.z
        var br: float = absf(axis.dot(b.basis.x))*half.x + absf(axis.dot(b.basis.y))*half.y + absf(axis.dot(b.basis.z))*half.z
        if absf(axis.dot(a.origin-b.origin)) >= ar + br - 0.0001:
            return false
    return true
