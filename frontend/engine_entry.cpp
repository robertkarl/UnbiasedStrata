// frontend/engine_entry.cpp - the unmodified Strata engine program, compiled into this binary under another name.
//
// The engine's own source file is included as it is; only its entry point is renamed, so frontend/main.cpp can
// decide whether this process is the HTTP front end or the engine it starts as a child (`--engine`).
#define main strata_engine_main
#include "../src/program/generate.cpp"
