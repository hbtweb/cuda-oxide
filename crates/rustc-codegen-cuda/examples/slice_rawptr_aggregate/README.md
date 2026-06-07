# slice_rawptr_aggregate

## Regression test — `&y0[1..]` subslice (RawPtr aggregate) import

A self-contained guard for an importer failure on subslicing a slice. A
`#[kernel]` takes a `&[u8]`, forms the subslice `&x0[1..]`, passes it to a
`#[device]` helper, and the host verifies the bytes read through the subslice.

## The bug it guards against

`&x0[1..]` does not lower to a simple pointer offset; it constructs a fresh fat
pointer via `Rvalue::Aggregate(AggregateKind::RawPtr(Slice(u8), mut),
[data_ptr, len])`. The MIR importer's `Aggregate` match had **no `RawPtr`
arm**, so it hit the catch-all and failed import with `Aggregate kind
RawPtr(...Slice...) not yet supported`. Op verification (`MirInsertFieldOp`)
also rejected a `MirSliceType` aggregate.

The fix:

1. Adds a `RawPtr` arm to the importer's `Aggregate` translation. For a `Slice`
   pointee it builds a `MirSliceType` fat pointer
   (`undef → insert ptr@0 → insert len@1`).
2. Relaxes `MirInsertFieldOp::verify` to accept a `MirSliceType` aggregate
   (field 0 = ptr to element, field 1 = integer len), mirroring the existing
   `MirSliceType` arm in `MirExtractFieldOp::verify`.

During lowering the slice-typed `MirUndefOp` becomes a non-opaque LLVM
`{ptr, i64}` struct, so the inserts go through the `is_lowered_llvm_struct`
path and emit `insertvalue` at 0/1 directly — no `mir-lower` change needed.

## Pre-fix vs post-fix

- **Pre-fix:** build fails `Aggregate kind RawPtr(...Slice...) not yet
  supported`.
- **Post-fix:** the kernel builds, runs, and `doot(&x0[1..])` reads the
  subslice correctly.

## Build and run

```bash
cargo oxide run slice_rawptr_aggregate
```

## Expected output

```text
=== slice_rawptr_aggregate: `doot(&x0[1..])` subslice on the GPU ===

slice_rawptr_aggregate: PASS (doot(&x0[1..]) = 13090 = 0x3322, expected 0x3322)
```

## Hardware requirements

- **Minimum GPU:** any CUDA-capable GPU
- **CUDA Driver:** 11.0+
