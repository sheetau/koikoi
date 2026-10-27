extends RefCounted
## Public card identities only. Rendering layout never sees a hidden deck.

const LIGHTS = [0, 8, 28, 40, 44]
const ANIMALS = [4, 12, 16, 20, 24, 29, 32, 36, 41]
const RIBBONS = [1, 5, 9, 13, 17, 21, 25, 33, 37, 42]

static func category(id: int) -> int:
	if id in LIGHTS:
		return 3
	if id in ANIMALS:
		return 1
	if id in RIBBONS:
		return 2
	return 0

static func field(index: int, count: int) -> Vector3:
	var columns := maxi(4, ceili(count / 2.0))
	var spacing := maxf(0.96, minf(1.32, 7.4 / (columns - 1)))
	var right := minf(2.8, -0.9 + (columns - 1) * spacing / 2)
	return Vector3(right - (columns - 1 - index / 2) * spacing, 0.012, -0.85 if index % 2 == 0 else 0.85)

static func field_left(count: int) -> float:
	return minf(-5.3, field(0, count).x - 0.75)

static func capture(id: int, slot: int, player: int, counts: Array = [9, 4, 4, 2]) -> Vector3:
	# Reserve the largest unscored group: nine chaff, four animals/ribbons,
	# and two dry lights plus rain. Scored groups leave for the yaku rows.
	var group := category(id)
	var left := -5.1
	var row := 0
	for kind in range(4):
		var width := 0.78 + maxf(0, maxi(counts[kind], [9, 4, 4, 3][kind]) - 1) * 0.39
		if left + width > 5.1:
			left = -5.1
			row += 1
		if kind == group:
			return Vector3(left + 0.39 + slot * 0.39, 0.014 + slot * 0.012, (1 if player == 0 else -1) * (3.65 + row * 1.5))
		left += width + 0.2
	return Vector3.ZERO
