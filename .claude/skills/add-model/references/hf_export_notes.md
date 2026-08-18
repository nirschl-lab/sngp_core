# Keeping a net HF-exportable

`scripts/hf/export_to_hub.py` produces a self-contained `trust_remote_code` bundle by
concatenating the *source* of `src/models/backbones.py`, `src/models/outputs.py`,
`src/models/components/spectral_norm.py` (SNGP only), and the net class itself —
never by importing `src.*` at runtime in the exported bundle.

For a new net to be exportable this way:

1. Its module must import nothing from `src.*` except `src.models.backbones`,
   `src.models.outputs`, `src.models.registry`, and (if using spectral norm)
   `src.models.components.spectral_norm`. No `hydra`, no `lightning`, no other `src.*`
   subpackage.
2. Add an entry to `_FAMILIES` in `scripts/hf/export_to_hub.py`: a builder function
   (model `_build_<family>_module_source()`, see the existing two for the pattern —
   `inspect.getsource()` the whole file for dependency-light modules, and just the
   class for the net class itself, stripping the `@register_net(...)` decorator).
3. If the net has any custom `nn.Module`-level constructs beyond what's already
   vendored (e.g. a new head with its own state), make sure its `__init__` doesn't
   reference anything outside its own module — `inspect.getsource()` on a class only
   captures that class's body, not helper functions defined elsewhere in the file.

If none of this applies (the net is training/research-only, not meant for
distribution), skip HF export entirely — it's optional, not required for a net to be
usable in this project.
