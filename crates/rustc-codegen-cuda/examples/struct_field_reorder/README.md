# struct_field_reorder

## Regression test — aggregate field-index lowering for reordered/padded `repr(Rust)` structs

A self-contained guard for a device-codegen miscompile in `mir-lower`'s
aggregate field-access lowering. It builds a generic `ScratchArena<S>` (from the
sibling `arena_core` crate) on the device, drives it through `&self` methods
across the crate boundary, and the host verifies every field landed at its
layout-computed index.

## The bug it guards against

`ScratchArena<S>` is a `repr(Rust)` struct whose fields rustc **reorders** in
memory and (because one field is an align-4, multi-word enum) **pads** with an
interior `[N x i8]` field. Mapping a declaration-order field index to the LLVM
struct field index was wrong in two ways:

1. `extract_field` / `insert_field` fell back to an **identity** mapping when the
   operand's `MirStructType` was missing from the conversion type-history —
   addressing the wrong field of a reordered struct.
2. `field_addr` / `construct_struct` honored the reorder but **not** the
   `[N x i8]` padding fields — indices were off by the padding count.

## Pre-fix vs post-fix

- **Pre-fix:** device codegen fails — LLVM module verification rejects an
  `insert_value` whose index lands on the struct's `[N x i8]` padding field
  (`Value being inserted / extracted does not match the type of the indexed
  aggregate`).
- **Post-fix:** the kernel builds, runs, and every field verifies.

So if the miscompile ever returns, this example stops building — a positive
regression guard.

## Why a separate `arena_core` crate

The bug only surfaces when the generic `&self` receiver's `MirStructType` is
reconstructed across a crate boundary at monomorphization. Keeping the arena in
its own crate reproduces that shape self-contained, with no external
dependencies. `opt-level = 0` keeps the optimizer from scalarizing the
reordered/padded struct into SSA values before lowering.

## Build and run

```bash
cargo oxide run struct_field_reorder
```

## Expected output

```text
=== struct_field_reorder: reordered/padded generic ScratchArena on the GPU ===

struct_field_reorder: PASS (cursor=256, all 256 slots' fields landed at the reordered/padded field indices)
```

## Hardware requirements

- **Minimum GPU:** any CUDA-capable GPU
- **CUDA Driver:** 11.0+
