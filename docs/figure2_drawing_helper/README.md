# Figure 2 drawing helper

This directory contains visual resources used to draw and communicate the
RelScope method architecture.

- `figure2_method_architecture.png` is the manuscript-facing Figure 2 preview.
- `relscope_figure_surface_generator.html` is an auxiliary interactive drawing
  tool for producing generic paired surfaces, SDF-style fields, critical
  regions, heat maps, and relational-pooling illustrations.

The HTML tool is independent of the RelScope training and inference pipeline.
It uses synthetic generic surfaces, contains no patient data, and does not
produce experimental results.

## Open the helper locally

The tool imports Three.js from a public CDN, so an internet connection is
needed when it is opened.

```bash
python -m http.server 8000 --directory docs/figure2_drawing_helper
```

Then open
[`http://localhost:8000/relscope_figure_surface_generator.html`](http://localhost:8000/relscope_figure_surface_generator.html).

The interface provides presets for paired structures, surface fields, critical
heat maps, relational pooling, and a generic tooth–nerve view. Exports can be
saved as PNG, SVG, or OBJ and then refined in a vector or slide editor.
