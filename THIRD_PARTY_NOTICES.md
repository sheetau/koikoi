# Third-party sources

The Koi-Koi rule specification, card classifications and compatible native300-feature extraction were derived from guansanghai/KoiKoi-AI, revision 89d789565c7c36c84b74662e43001d960bf80901, https://github.com/guansanghai/KoiKoi-AI. The optional adapter imports its unmodified code and published weights from an ignored external checkout. The experimental differentiable PublishedActor also reconstructs these original modules and records the published weights as its initialization dependency. The following license is retained for these sources and derived rule/feature implementations.

The local human-play GUI reads card PNG images from that same pinned external checkout at runtime. These images are not copied into this repository; retain the upstream attribution and license with the external files.

MIT License

Copyright (c) 2023 Sanghai Guan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Floating-table game assets

The 48 PNG images in `game/assets/cards/` are the traditional-color Hanafuda
set by Louie Mantia and すけじょ, from Wikimedia Commons:
https://commons.wikimedia.org/wiki/Category:SVG_Hanafuda_with_traditional_colors_(black_border) .
They are licensed under Creative Commons Attribution-ShareAlike 4.0 International:
https://creativecommons.org/licenses/by-sa/4.0/ . Original downloaded PNG bytes, per-file links
and checksums are retained; see `game/assets/cards/SOURCE.md`, `sources.json`,
`manifest.json` and `LICENSE.txt`. The client masks only the outer frame with
#b44519 while preserving the illustration. The adapted appearance remains under
CC BY-SA 4.0. This asset license applies to the artwork.

`game/assets/fonts/NotoSansJP.ttf` is Noto Sans JP from the Noto Project via
https://github.com/google/fonts/tree/main/ofl/notosansjp . The SIL Open Font
License 1.1 is retained in `game/assets/fonts/OFL.txt`.
Font SHA256: `c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f`.

The external Godot 4.7.2 engine is MIT-licensed; see
https://godotengine.org/license/ and its distribution notices. The setup script
verifies the pinned release archive against its SHA512 before extraction.

`game/assets/fonts/NotoSerifJP-Black.otf` is Noto Serif JP Black, from
https://github.com/notofonts/noto-cjk/tree/main/Serif/SubsetOTF/JP .
The SIL Open Font License 1.1 is retained in `game/assets/fonts/NotoSerif-OFL.txt`.

`game/assets/fonts/AntiqueSix-ExtraBold.ttf` was supplied by the user and copied
into this project at their request. Its embedded copyright is Copyright (c)
2005 by Jordan Davies. This font is separate from the Noto OFL license; the
supplied file has no embedded license grant.

The six MP3 clips in `game/assets/audio/` were supplied by the user and copied
unchanged from their concepts directory. Original authorship and license
information were not supplied. The four WAV UI cues are original procedural
pulse/triangle effects created for this project, with no sampled source audio.
See `game/assets/audio/SOURCE.txt` for assignments.
