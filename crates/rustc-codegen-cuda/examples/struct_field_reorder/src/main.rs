/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for the aggregate field-index lowering fix.
//!
//! A `#[kernel]` builds a generic `ScratchArena<S>` (from the sibling
//! `arena_core` crate) and drives it through `&self` methods on the device. The
//! arena is a `repr(Rust)` struct whose fields rustc REORDERS in memory (a
//! generic fat-pointer store, a `Layout` enum, and `u32`/smaller scalars), so
//! the declaration index never equals the LLVM struct field index and the
//! lowered struct carries `[N x i8]` padding. The fix this guards corrects two
//! defects in `extract_field` / `insert_field` / `field_addr` /
//! `construct_struct`:
//!
//!   1. They fell back to an IDENTITY (declaration-order) index when the
//!      operand's `MirStructType` was not in the type-history, addressing the
//!      wrong field of a reordered struct.
//!   2. The index walk ignored the `[N x i8]` PADDING fields the type converter
//!      inserts for explicit layouts, so later fields were off by padding.
//!
//! The host verifies that every slot's two fields landed at the
//! `Layout::Soa`-computed indices, so a mis-mapped field index is observable.
//!
//! Pre-fix this fails device codegen: LLVM module verification rejects an
//! `insert_value` whose index lands on the struct's `[N x i8]` padding field.
//! Post-fix it builds and every field verifies.
//!
//! Build and run with:
//!   cargo oxide run struct_field_reorder

use arena_core::{alloc_and_fill, CellStore, Layout, ScratchArena};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::{cuda_module, kernel, thread};

/// Device-backed cell store: an atomic bump cursor plus a raw data buffer.
/// `data` is `&[u32]` but aliases a writable device buffer (the same raw-write
/// trick var-mem's GPU example uses).
pub struct DeviceStore<'a> {
    pub cursor: &'a DeviceAtomicU32,
    pub data: &'a [u32],
}

impl CellStore for DeviceStore<'_> {
    #[inline]
    fn bump(&self, n: u32) -> u32 {
        self.cursor.fetch_add(n, AtomicOrdering::Relaxed)
    }
    #[inline]
    fn store(&self, idx: u32, val: u32) {
        unsafe {
            let p = self.data.as_ptr() as *mut u32;
            *p.add(idx as usize) = val;
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fill(cursor: &[u32], data: &[u32], params: &[u32]) {
        let cap = params[0];
        let stride = params[1];

        let cur = unsafe { &*(cursor.as_ptr() as *const DeviceAtomicU32) };
        let store = DeviceStore { cursor: cur, data };
        let arena = ScratchArena::new(store, cap, stride, Layout::Soa);

        // Drive the arena through a `&ScratchArena<S>` receiver across the
        // (non-inlined) crate boundary. Two fields per slot: field 0 = slot,
        // field 1 = slot*10. Under Layout::Soa the index is field*cap+slot, so
        // a wrong field index (the bug) lands the value in the wrong half of
        // `data`.
        alloc_and_fill(&arena);

        let _ = thread::index_1d();
    }
}

fn main() {
    println!("=== struct_field_reorder: reordered/padded generic ScratchArena on the GPU ===\n");

    const CAP: u32 = 256;
    const STRIDE: u32 = 2;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let cursor = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    let data = DeviceBuffer::<u32>::zeroed(&stream, (CAP * STRIDE) as usize).unwrap();
    let params = DeviceBuffer::from_host(&stream, &[CAP, STRIDE]).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .fill(
            &stream,
            LaunchConfig::for_num_elems(CAP),
            &cursor,
            &data,
            &params,
        )
        .expect("Kernel launch failed");

    let cur_final = cursor.to_host_vec(&stream).unwrap()[0];
    let data = data.to_host_vec(&stream).unwrap();

    let mut errors = 0usize;
    if cur_final != CAP {
        eprintln!("  cursor: got {cur_final}, expected {CAP}");
        errors += 1;
    }

    // Layout::Soa index(slot, field) = field*CAP + slot.
    for slot in 0..CAP {
        let v0 = data[slot as usize];
        let v1 = data[(CAP + slot) as usize];
        let ok = v0 == slot && v1 == slot.wrapping_mul(10);
        if !ok {
            if errors < 8 {
                eprintln!(
                    "  slot {slot}: field0 {}/{} field1 {}/{}",
                    v0,
                    slot,
                    v1,
                    slot.wrapping_mul(10),
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!(
            "struct_field_reorder: PASS (cursor={cur_final}, all {CAP} slots' fields \
             landed at the reordered/padded field indices)"
        );
    } else {
        println!("\nstruct_field_reorder: FAILED: {errors} mismatches");
        std::process::exit(1);
    }
}
