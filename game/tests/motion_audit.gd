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
        max_lift = maxf(max_lift, card.lift + (Array(card.clearance).max() if not card.clearance.is_empty() else 0))
        if not card.orbit:
            normal_max = maxf(normal_max, card.lift + (Array(card.clearance).max() if not card.clearance.is_empty() else 0))
        var h: Vector3 = table.half_bounds(card.basis)
        if absf(card.position.x) < 5.3 - h.x and absf(card.position.z) < 2.85 - h.z and card.position.y - h.y < -0.002:
            board_hits += 1
        for other in table.bodies:
            if other == card or not other.visible:
                continue
            if intersects(card.transform, other.transform):
                overlaps += 1
                if not cases.has(str(card.identity) + ":" + str(other.identity)) and cases.size() < 8:
                    cases[str(card.identity) + ":" + str(other.identity)] = true
                    push_error("Card intersection: %s / %s at flight t=%.3f" % [card.identity, other.identity, card.elapsed / card.duration])
func intersects(a: Transform3D, b: Transform3D) -> bool:
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
