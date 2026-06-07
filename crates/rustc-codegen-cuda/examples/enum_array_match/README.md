# enum_array_match

## Regression test — enum-in-array `match` projection import

A self-contained guard for an importer failure on `match xs[i]` where `xs` is an
array (or slice) of an enum. A `#[kernel]` builds a device-local `[E; 4]`,
matches an element by both a runtime and a constant index, and the host verifies
every match arm.

## The bug it guards against

`match xs[i] { E::A(x) => ..., E::B(y) => ..., E::C => ... }` lowers to a MIR
projection chain `[Index, Downcast, Field]` (or `[ConstantIndex, Downcast,
Field]` for a literal index). The importer's iterative place walker
(`translate_place_iterative`) advanced the projected SSA **value** in the
`Index` / `ConstantIndex` arms but never narrowed the running **Rust type**,
which stayed at the outer `Array(Adt(E), N)`. The following `Downcast` / `Field`
step then handed that array type to the enum-payload projection, which rejected
it with `Downcast on non-ADT type: Array`.

The fix narrows the running Rust type to the array/slice element type in both
index arms, so the `Downcast`/`Field` step resolves against `Adt(E)`.

## Pre-fix vs post-fix

- **Pre-fix:** import fails with `Downcast on non-ADT type: Array`.
- **Post-fix:** the kernel builds, runs, and every match arm verifies.

## Build and run

```bash
cargo oxide run enum_array_match
```

## Expected output

```text
=== enum_array_match: `match xs[i]` over an enum array on the GPU ===

  match xs[0]: PASS (result = 7)
  match xs[1]: PASS (result = 1008)
  match xs[2]: PASS (result = 9999)
  match xs[3]: PASS (result = 100)
  match xs[0] (const): PASS (result = 7)

enum_array_match: PASS (all match arms over the enum array verified)
```

## Hardware requirements

- **Minimum GPU:** any CUDA-capable GPU
- **CUDA Driver:** 11.0+
