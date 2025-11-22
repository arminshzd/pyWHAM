# Repository Guidelines

## Project Structure & Module Organization
- `wham/`: single-dimension WHAM implementation producing the `wham` CLI; depends on `nr/ran2.c` and `nr/locate.c`.
- `wham-2d/`: two-dimensional variant producing the `wham-2d` CLI.
- `nr/`: Numerical Recipes utilities shared by both targets.
- `doc/`: LaTeX source and PDF usage notes; update `doc.tex` when changing CLI behavior.
- `make/Makefile`: legacy wrapper that builds both subdirectories; `CMakeLists.txt` supports the modern CMake flow; `mk_dist` archives the tree for distribution.
- Build outputs land in `build/` when using CMake or within each component directory when using the legacy Makefiles.

## Build, Test, and Development Commands
- `cmake -S . -B build` then `cmake --build build` — configure and compile both binaries with the flags defined in CMake.
- `cmake --install build --prefix <path>` — install the binaries to a prefix (defaults to system locations if omitted).
- `make -C make` — legacy make path that invokes the nested Makefiles for `wham` and `wham-2d`.
- `make -C wham clean` / `make -C wham-2d clean` — remove objects and binaries from legacy builds.
- Run locally: `./build/wham <args>` or `./build/wham-2d <args>`; see `doc/doc.pdf` for metadata/histogram file formats and CLI usage examples.

## Coding Style & Naming Conventions
- C code with 4-space indentation, no tabs; stick to the existing C99-compatible patterns in the codebase.
- Functions and variables use lower_snake_case; macros and constants stay uppercase; keep public prototypes in the module headers (`wham.h`, `wham-2d.h`).
- Use `static` for file-local helpers and check return codes for file I/O and allocations.
- Maintain the current logging/printf patterns so downstream scripts parsing CLI output remain stable.

## Testing Guidelines
- No automated test suite exists; validate changes by rebuilding and running on representative metadata/histogram files (include the command lines in PRs).
- When altering numerics, compare free energy or probability outputs against a known good run and note acceptable tolerances.
- Enable extra warnings for debugging: `CFLAGS="-g -Wall -Wextra" cmake -S . -B build && cmake --build build`.

## Commit & Pull Request Guidelines
- Commits: short, imperative subjects (<72 chars) with rationale in the body when touching algorithms or I/O.
- PRs: describe input/output changes, list commands run (builds + manual checks), and mention any documentation updates; link issues or tickets when available.
- Exclude generated binaries, archives (e.g., `wham`, `wham-2d`, `wham-dist.tar.gz`), and large dataset inputs from commits.

## Security & Configuration Tips
- The CLI consumes user-provided metadata; validate file paths and propagate errors rather than silently truncating them.
- Randomized bootstrap runs should log seeds when determinism is needed for reproducibility.
