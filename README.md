# taskflow-musa

`taskflow-musa` is the Moore Threads MUSA maintenance line of
[Taskflow](https://github.com/taskflow/taskflow), a C++ task-parallel and
heterogeneous workflow library.

## Source and version

- Upstream project: Taskflow
- Inherited version: `v4.1.0`
- Upstream baseline: `c312d58dfb98e17e6d59d539007f7a69f9c3ea07`
- Maintainer: Moore Threads

The MUSA line adds MUSA examples and backend integration while retaining the
upstream C++ API, CUDA support, CPU tasking, examples, tests, and third-party
notices.

## Requirements and build

Use CMake 3.18 or newer and a C++17-capable compiler. Install the MUSA SDK when
selecting the MUSA backend. A typical build is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

Enable examples, tests, and accelerator backends using the options documented
by the project. Toolkits and optional libraries are external prerequisites.

## Layout and contribution

The header library is under `taskflow/`; examples, modules, tests, and CMake
files retain the upstream layout. Keep backend changes isolated and preserve
the public API and component notices.

Moore Threads maintains this source line. See [LICENSE](LICENSE) for the
applicable declaration.
