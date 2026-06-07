# option_ref_unwrap_or

## Regression test — `Option<&T>` reference-to-scalar constant import

A self-contained guard for a silent miscompile (CUDA 700) when rustc
const-folds `None.unwrap_or(&77)` to a reference-to-scalar constant. A
`#[kernel]` builds a `Some(&r)`/`None` pair of `Option<&u32>` and writes
`*a.unwrap_or(&77)` and `*b.unwrap_or(&77)`; the host verifies the values.

## The bug it guards against

With both a `Some` and a `None` of `Option<&u32>` live in one frame, rustc -O
const-folds `None.unwrap_or(&77)` to a reference-to-**scalar** constant `&77` —
a `ConstantKind::Allocated` whose 8 pointer bytes are a zero relocation
placeholder, with the real `77` allocation recorded in
`alloc.provenance.ptrs`.

The MIR importer's pointer-constant arm in `translate_operand` followed
provenance **only** when the pointee was a struct (`translate_struct_constant`
+ `mir.ref`). For a scalar pointee it skipped that and fell through to the
"raw pointer constant" branch, which emitted `inttoptr i64 0` — dropping the
relocation. The later deref then became `load i32, ptr null` => **CUDA 700**
(illegal memory access) at runtime, with no diagnostic.

The fix broadens the gate from "pointee is a struct with data" to "constant has
provenance" and dispatches:

- struct pointee → existing `translate_struct_constant` + `mir.ref` path;
- scalar pointee → new path that follows the provenance target allocation's
  bytes (`follow_provenance_target_bytes`), materializes the pointee value via
  `translate_constant_value_from_bytes`, and wraps it in a `MirRefOp`.

## Pre-fix vs post-fix

- **Pre-fix:** `*b.unwrap_or(&77)` derefs address 0 → `DriverError(700, "an
  illegal memory access was encountered")`.
- **Post-fix:** the kernel runs and `out[0] == 5`, `out[1] == 77`.

## Build and run

```bash
cargo oxide run option_ref_unwrap_or
```

## Expected output

```text
=== option_ref_unwrap_or: None.unwrap_or(&77) reference-to-scalar constant ===

option_ref_unwrap_or: PASS (out[0]=5 expected 5, out[1]=77 expected 77)
```

## Hardware requirements

- **Minimum GPU:** any CUDA-capable GPU
- **CUDA Driver:** 11.0+
