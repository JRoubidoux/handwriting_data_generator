Mark1:
    Not much here as I didn't do this yet in the project. 



Mark2:
    On the topic of Font files. It is the case that font files (.ttf or .otf) have a character map that has a few different tables that contain information about character encodings. The character codes for these tables point to a glyph that contains information about how to render a font image. 

        Font File (.ttf / .otf)
        ├── cmap (Character-to-Glyph Mapping)
        │    ├── Subtable 1 (Platform ID 0, Encoding ID 3) - Unicode BMP (Format 4)
        │    │    ├── Character code U+0041 -> Glyph ID 3 (for 'A')
        │    │    └── Character code U+0061 -> Glyph ID 4 (for 'a')
        │    ├── Subtable 2 (Platform ID 3, Encoding ID 1) - Microsoft Symbol (Format 0)
        │    │    └── Character code 33 -> Glyph ID 5 (for '!')
        │    └── Subtable 3 (Platform ID 1, Encoding ID 0) - Macintosh Roman (Format 0)
        │         └── Character code 65 -> Glyph ID 3 (for 'A')
        ├── glyf (Glyph Outline Table)
        │    ├── Glyph ID 3 -> Outline (quadratic Bézier curves for 'A')
        │    └── Glyph ID 4 -> Outline (quadratic Bézier curves for 'a')
        └── head, hhea, maxp, name, OS/2, post, and other tables
