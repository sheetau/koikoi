# Traditional-color Hanafuda

The 48 bundled PNG files are the **traditional color (black border)** set by
Louie Mantia and すけじょ, from Wikimedia Commons. They replace the previous
dotty-dev recoloring.

- [Collection](https://commons.wikimedia.org/wiki/Category:SVG_Hanafuda_with_traditional_colors_(black_border))
- [Requested example: March Hikari](https://commons.wikimedia.org/wiki/File:Hanafuda_March_Hikari_Alt.svg)
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

`sources.json` lists every original SVG file page and source URL. `manifest.json`
records the SHA256 of the downloaded bytes. The official 960-pixel-wide PNG
renderings are retained byte-for-byte.
At rendering time, `game/card_face.gdshader` replaces only the outer rectangular
frame with #b44519; black illustration ink inside the frame remains unchanged.
The adapted appearance is also provided under CC BY-SA 4.0. The license applies
to the artwork, not the game's code.

To refresh the cached/bundled images from the pinned source URLs:

```powershell
./.venv/Scripts/python.exe -X utf8 -m scripts.fetch_game_cards
```

The downloader identifies this project, spaces requests, caches images in
ignored `external/traditional-hanafuda/`, and stops on HTTP errors. If Commons
returns 429, honor its Retry-After interval before retrying. Playing the game
uses only the bundled assets and does not contact Commons.
