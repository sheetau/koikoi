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

The 48 unmodified SVG images in `game/assets/cards/` are Louie Mantia's
Hanafuda deck, recolored by dotty-dev, from
https://github.com/dotty-dev/Hanafuda-Louie-Recolor at commit
`a4ac60e4bd60f95ab8c4e5176b50bfd2e3772992`.
Licensed under Creative Commons Attribution-ShareAlike 4.0 International:
https://creativecommons.org/licenses/by-sa/4.0/ . Full license and original
source-image links are retained in `game/assets/cards/LICENSE.txt` and
`game/assets/cards/SOURCE.md`. Godot generates raster caches/mipmaps from the
original SVGs for display. This asset license applies to the artwork.

`game/assets/fonts/NotoSansJP.ttf` is Noto Sans JP from the Noto Project via
https://github.com/google/fonts/tree/main/ofl/notosansjp . The SIL Open Font
License 1.1 is retained in `game/assets/fonts/OFL.txt`.
Font SHA256: `c2f3b4d463500a2ddcd3849cded1fceeb9fd6d1c32e6cbecd568453ba50fc68f`.

The external Godot 4.7.2 engine is MIT-licensed; see
https://godotengine.org/license/ and its distribution notices. The setup script
verifies the pinned release archive against its SHA512 before extraction.
