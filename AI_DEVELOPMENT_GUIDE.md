# AI Development Guide for PopSift

This guide defines how AI-assisted code generation should be done in this repository.  
It ensures that contributions (from GitHub Copilot, ChatGPT, Claude, etc.) follow a **consistent, modern, and maintainable style**.

---

## General Principles

- Always prioritize **readability** and **clarity** over micro-optimizations.
- Follow **modern C++17 best practices**.
- Keep device-side __global__ functions in the same source file as the host-side C++ code that starts this kernel.
- Always compile __device__ functions with the functions that call them. Preferably declare them static inline.
- Prefer **modularity**: each class or major component should live in its own file.
- Code should be **self-documenting** whenever possible, with clear naming and structure.

---

## C++ Guidelines

- **Standard**:
  - Use **C++17**. Prefer `constexpr`, `auto` and `enum class`.
  - Use range-based for loops on the host side.
  - Use smart pointers (`std::unique_ptr`, `std::shared_ptr`) on the host side.
  - Never pass smart pointers as parameters to __global__ functions.
  - Avoid dynamic memory allocation on the device side.
- **Memory Management**: Use RAII. Avoid raw `new`/`delete` except in CUDA contexts where unavoidable.
- **Error Handling**:
  - Use exceptions in host C++ code.  
  - In CUDA, check and propagate error codes using helper utilities/macros. Never ignore errors.
- **Namespaces**: Group related functions/classes logically. Avoid polluting the global namespace.
- **Headers**:
  - Keep headers minimal; forward declare instead of including heavy dependencies.
  - Each header should be guarded with `#pragma once`.
- **Style**:
  - `snake_case` for variables and functions.  
  - `CamelCase` for class and struct names.  
  - `ALL_CAPS` for macros and compile-time constants.

---

## CUDA Guidelines

- Separate **kernels** from host orchestration code.
- Name kernels descriptively, e.g. `compute_gradient_kernel`.
- Document assumptions about:
  - Thread/block layout
  - Shared memory usage
  - Synchronization requirements
- Use `__restrict__` and `constexpr` where appropriate for performance and clarity.
- Prefer small, focused kernels over overly complex ones.
- Always validate CUDA API calls.

---

## Threading Guidelines

- **Host Threading**: Use `std::thread` and synchronization primitives from `<mutex>`.
- **CUDA Streams**: Use multiple streams for concurrent kernel execution.
- **Thread Safety**: Document thread safety guarantees for all public APIs.
- **Avoid**: Raw pthreads or platform-specific threading APIs.

---

## Modularity and Organization

- Keep code **organized by functionality** (e.g., detection, description, GPU utilities).
- Avoid very long functions (>50 lines); refactor into helpers when possible.
- Prefer **free functions** in namespaces over singletons or unnecessary wrapper classes.
- Keep algorithms and data structures reusable when possible.

---

## Performance Guidelines

- **Memory Access Patterns**: Prefer coalesced memory access in CUDA kernels. Document stride patterns.
- **Shared Memory**: Use shared memory for data reuse within thread blocks. Document bank conflicts.
- **Register Usage**: Monitor register pressure in kernels. Aim for high occupancy.
- **Asynchronous Operations**: Use CUDA streams for overlapping computation and memory transfers.
- **Profiling**: Profile with `nvprof` or Nsight before optimizing. Document performance assumptions.
- **Memory Bandwidth**: Consider memory bandwidth as the primary bottleneck for most kernels.

---

## Documentation

- Use **Doxygen-style comments** for public APIs, classes, and CUDA kernels.
- Document algorithm choices and any CUDA-specific design tradeoffs.
- Update examples and README when new features are introduced.
- At each update ensure that the changelog is also updated following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.
  - for each new feature, bug fix, or breaking change, add a corresponding entry in the changelog.
  - the description should be short but informative, followed by the relevant PR link.

---

## Git Guidelines

- **Branch Names**: `feature/description`, `fix/issue-number`, `refactor/component`
- **Commit Messages**: Use conventional commits format: `[feat]`, `[fix]`, `[refactor]`, `[doc]` etc.
- **File Organization**: Keep related files in logical directories
- **Ignore Patterns**: Update `.gitignore` for build artifacts and IDE files

---

## Commit & PR Guidelines

- Keep commits small and focused (one feature or fix per commit).
- Do not commit untracked files that are not relevant.
- PRs should include:
  - Clear description of changes
  - Explanations for algorithmic choices or CUDA-specific design decisions
  - Updated tests or examples if applicable
- Code must pass existing CI checks before merging.
