# ref_array_elem_write

## Regression test — `&mut (*arr)[i]` write through a referenced array element

A self-contained guard for a silent dropped-write miscompile. A `#[kernel]`
takes a `&mut [f32; N]`, writes `42.0` through `&mut arr[i]` to every element,
copies the array to `out`, and the host verifies every element is `42.0`.

## The bug it guards against

Taking `&mut arr[i]` where `arr: &mut [f32; N]` lowers to
`Rvalue::Ref(place = [Deref, Index])`. The importer's address helper
`translate_place_addr_from_slot` punted on a leading `Deref` (its catch-all
returned `Ok(None)`), so the `Rvalue::Ref` arm fell through to Case 5 — the
`MirRefOp` fallback. That path materializes the element as a **loaded value**
and stores the **copy** in a fresh stack slot, so `*e = 42.0` writes the copy,
not the array. The write is silently dropped (the array element stays `0`), with
no diagnostic.

The fix teaches `translate_place_addr_from_slot` to address through:

- **`Deref`** — load the inner pointer (the result must stay a pointer so the
  next projection can address into the pointee, e.g. the `&mut [f32; N]` local's
  slot is `*mut *mut [f32; N]`; the load yields `*mut [f32; N]`).
- **runtime `Index`** — when the pointer points at an array, load the index
  local and emit `MirArrayElementAddrOp` (mirroring the existing `ConstantIndex`
  arm), yielding `*mut f32` into the real array.

So the address is computed into the original array and the write persists.

## Pre-fix vs post-fix

- **Pre-fix:** every element stays `0` (the write is dropped).
- **Post-fix:** every element is `42.0`.

## Build and run

```bash
cargo oxide run ref_array_elem_write
```

## Expected output

```text
=== ref_array_elem_write: &mut (*arr)[i] write through a referenced element ===

ref_array_elem_write: PASS (all 8 elements written through &mut arr[i] == 42.0)
```

## Hardware requirements

- **Minimum GPU:** any CUDA-capable GPU
- **CUDA Driver:** 11.0+
