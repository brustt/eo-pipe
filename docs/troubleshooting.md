# Troubleshooting

## GDAL: `No module named '_gdal'` / `undefined symbol: CPLQuietWarningsErrorHandler`

### Symptom

```
ImportError: .../osgeo/_gdal.cpython-312-x86_64-linux-gnu.so:
    undefined symbol: CPLQuietWarningsErrorHandler
ModuleNotFoundError: No module named '_gdal'
```

### Cause

Two GDAL C libraries coexist on the system:

| Source | Library | Version |
|---|---|---|
| apt (system) | `/lib/x86_64-linux-gnu/libgdal.so.34` | 3.8.x |
| manual install | `/usr/local/lib/libgdal.so.38` | 3.12.x |

When the Python `gdal` extension is compiled, the linker picks up the apt
library (`libgdal.so.34`) because `/lib/x86_64-linux-gnu` appears before
`/usr/local/lib` in the default search path.  The resulting `.so` is built
against the 3.12 headers (which declare newer symbols) but linked to the 3.8
library (which doesn't have them) — runtime crash.

### Fix: rebuild with the correct linker path

Run this once after creating or recreating the venv:

```bash
uv pip install setuptools numpy
LDFLAGS="-L/usr/local/lib -Wl,-rpath,/usr/local/lib" \
  uv pip install --no-build-isolation --reinstall "gdal==3.12.2"
```

`-Wl,-rpath,/usr/local/lib` bakes the path to `libgdal.so.38` directly into
the extension so it loads the right library regardless of `LD_LIBRARY_PATH`.

Verify with:

```bash
uv run python -c "from osgeo import gdal; print(gdal.__version__)"
# expected: 3.12.2
```
