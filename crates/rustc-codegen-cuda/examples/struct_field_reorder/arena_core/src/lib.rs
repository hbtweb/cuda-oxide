/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `arena_core` — a self-contained `no_std` generic bump arena.
//!
//! This is a minimal, dependency-free shrink of var-mem-core's `ScratchArena`,
//! kept in its own crate so the GPU example that depends on it reproduces the
//! aggregate field-index lowering bug. The essential ingredients:
//!
//!   * `ScratchArena<S>` is a `repr(Rust)` generic struct whose fields rustc
//!     REORDERS in memory: a generic `store: S` (a fat-pointer-bearing
//!     aggregate), a `Layout` enum, and `u32`s. Declaration index never equals
//!     the LLVM struct field index, and the lowered struct carries `[N x i8]`
//!     padding fields.
//!   * The `&self` methods (`alloc` / `write`) access those reordered fields.
//!     Because the methods are monomorphized across the crate boundary, the
//!     receiver's `MirStructType` is reconstructed from the pointer pointee at
//!     lowering time — the exact path where the buggy identity fallback fired.
//!
//! Pre-fix the field accesses lower with the wrong index; post-fix they are
//! correct.

#![cfg_attr(not(test), no_std)]

/// Field-major vs slot-major addressing.
#[derive(Copy, Clone)]
pub enum Layout {
    /// Array of Structures: `slot * stride + field`.
    Aos,
    /// Structure of Arrays: `field * cap + slot`.
    Soa,
    /// Chunked layout with a chunk size. Payload variant: makes `Layout` an
    /// align-4, multi-word enum (like var-mem `Layout::AoSoA{chunk}`) so the
    /// arena struct's fields reorder and carry interior `[N x i8]` padding.
    AoSoA(u32),
}

impl Layout {
    /// Physical element index for `(slot, field)` under this layout.
    #[inline]
    pub fn index(self, slot: u32, field: u32, stride: u32, cap: u32) -> u32 {
        match self {
            Layout::Aos => slot.wrapping_mul(stride).wrapping_add(field),
            Layout::Soa => field.wrapping_mul(cap).wrapping_add(slot),
            Layout::AoSoA(chunk) => {
                let c = if chunk == 0 { 1 } else { chunk };
                let block = slot / c;
                let within = slot % c;
                block.wrapping_mul(stride).wrapping_mul(c)
                    .wrapping_add(field.wrapping_mul(c))
                    .wrapping_add(within)
            }
        }
    }
}

/// Backing store: an atomic bump cursor plus addressable cells.
pub trait CellStore {
    /// Atomically add `n` to the cursor; return the previous value.
    fn bump(&self, n: u32) -> u32;
    /// Store `val` at cell `idx`.
    fn store(&self, idx: u32, val: u32);
}

/// A generic bump arena over a [`CellStore`].
///
/// Declaration order is `store, cap, stride, layout`. rustc lays the fields out
/// in alignment-descending order — the generic fat-pointer `store` first, then
/// the `u32`s, then the single-byte `Layout` enum — so the declaration index
/// never matches the memory/LLVM field index, and the lowered LLVM struct
/// carries `[N x i8]` padding fields after the enum.
pub struct ScratchArena<S: CellStore> {
    store: S,
    cap: u32,
    stride: u32,
    layout: Layout,
}

impl<S: CellStore> ScratchArena<S> {
    #[inline]
    pub fn new(store: S, cap: u32, stride: u32, layout: Layout) -> Self {
        Self { store, cap, stride, layout }
    }

    /// Bump-allocate one slot, or `None` at capacity. Reads `store` and `cap`
    /// off `&self`.
    #[inline]
    pub fn alloc(&self) -> Option<u32> {
        let s = self.store.bump(1);
        if s >= self.cap { None } else { Some(s) }
    }

    /// Write `val` at `(slot, field)`. Reads `layout`, `stride`, `cap`, `store`
    /// off `&self` — the reordered field accesses under test.
    #[inline]
    pub fn write(&self, slot: u32, field: u32, val: u32) {
        let i = self.layout.index(slot, field, self.stride, self.cap);
        self.store.store(i, val);
    }
}

/// Allocate one slot and write its two fields, driving the arena through a
/// `&ScratchArena<S>` receiver across a `#[inline(never)]` crate boundary. This
/// keeps the receiver a pointer-to-struct (not scalarized into SSA values), so
/// the field accesses lower as `field_addr` / `extract_field` on the reordered,
/// padded `MirStructType` — the path the field-index fix corrects.
#[inline(never)]
pub fn alloc_and_fill<S: CellStore>(arena: &ScratchArena<S>) {
    if let Some(slot) = arena.alloc() {
        arena.write(slot, 0, slot);
        arena.write(slot, 1, slot.wrapping_mul(10));
    }
}
