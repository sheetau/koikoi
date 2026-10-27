extends RefCounted
var max_lift := 0.0
var normal_max := 0.0
var overlaps := 0
var board_hits := 0
var cases := {}
func inspect(table) -> void:
    for card in table.bodies:
        if not card.flight:
            continue
        var height: float = card.lift + (Array(card.clearance).max() if not card.clearance.is_empty() else 0)
        if height > max_lift and height > 5:
            print("MOTION_PEAK %.3f id=%s return=%s role=%s from=%s to=%s" % [height, card.identity, card.returning, card.role_route, card.flight_from, card.flight_to])
        max_lift = maxf(max_lift, height)
        if not card.orbit:
            normal_max = maxf(normal_max, card.lift + (Array(card.clearance).max() if not card.clearance.is_empty() else 0))
        var h: Vector3 = table.half_bounds(card.basis)
        if card.position.x > table.board_left + h.x and card.position.x < 5.3 - h.x and absf(card.position.z) < 2.85 - h.z and card.position.y - h.y < -0.002:
            board_hits += 1
        for other in table.bodies:
            if other == card or not other.visible:
                continue
            if intersects(card.transform, other.transform):
                overlaps += 1
                if not cases.has(str(card.identity) + ":" + str(other.identity)) and cases.size() < 8:
                    cases[str(card.identity) + ":" + str(other.identity)] = true
                    push_error("Card intersection: %s / %s at flight t=%.3f return=%s role=%s pos=%s/%s from=%s to=%s other_from=%s other_to=%s elapsed=%s/%s duration=%s/%s" % [card.identity, other.identity, card.elapsed / card.duration, card.returning, card.role_route, card.position, other.position, card.flight_from, card.flight_to, other.flight_from, other.flight_to, card.elapsed, other.elapsed, card.duration, other.duration])
func intersects(a: Transform3D, b: Transform3D) -> bool:
    # Exact enclosing AABBs cheaply reject separated pairs before the OBB SAT.
    # This only rejects impossible intersections; the narrow-phase tolerance stays unchanged.
    var half_box := Vector3(0.39, 0.64, 0.001)
    var a_extent := a.basis.x.abs() * half_box.x + a.basis.y.abs() * half_box.y + a.basis.z.abs() * half_box.z
    var b_extent := b.basis.x.abs() * half_box.x + b.basis.y.abs() * half_box.y + b.basis.z.abs() * half_box.z
    var separation := (a.origin - b.origin).abs() - a_extent - b_extent
    if separation.x > 0.0001 or separation.y > 0.0001 or separation.z > 0.0001:
        return false
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
