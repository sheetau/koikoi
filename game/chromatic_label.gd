extends Label3D
var echoes: Array[Label3D] = []
func _ready() -> void:
	for side in [-1,1]:
		var echo := Label3D.new()
		echo.name = "ChromaticEcho"
		echo.modulate = Color(1,0.22,0.12,0.42) if side<0 else Color(0.1,0.8,1,0.42)
		echo.outline_size = 0
		echo.position = Vector3(side*0.007,0,-0.001)
		add_child(echo)
		echoes.append(echo)
	_sync()
func _process(_delta: float) -> void:
	_sync()
func _sync() -> void:
	for echo in echoes:
		echo.text = text
		echo.font = font
		echo.font_size = font_size
		echo.pixel_size = pixel_size
		echo.double_sided = double_sided
		echo.horizontal_alignment = horizontal_alignment
		echo.vertical_alignment = vertical_alignment
		echo.texture_filter = texture_filter
		echo.layers = layers
