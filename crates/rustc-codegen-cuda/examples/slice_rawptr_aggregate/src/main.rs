/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for the `&y0[1..]` slice (RawPtr aggregate) import fix.
//!
//! Subslicing a slice — `&x0[1..]` — does not lower to a simple pointer offset;
//! it builds a fresh fat pointer via
//! `Rvalue::Aggregate(AggregateKind::RawPtr(Slice(u8), mut), [data_ptr, len])`.
//! The MIR importer's `Aggregate` match had no `RawPtr` arm, so it hit the
//! catch-all and failed import with `Aggregate kind RawPtr(...Slice...) not yet
//! supported`. (Op verification also rejected the `MirSliceType` aggregate in
//! `MirInsertFieldOp::verify`.)
//!
//! The fix adds a `RawPtr` arm that, for a `Slice` pointee, builds a
//! `MirSliceType` fat pointer (`undef → insert ptr@0 → insert len@1`), and
//! relaxes `MirInsertFieldOp::verify` to accept a `MirSliceType` aggregate.
//!
//! Pre-fix this FAILS to build. Post-fix it builds and the host verifies that
//! `doot(&x0[1..])` reads the subslice correctly.
//!
//! Build and run with:
//!   cargo oxide run slice_rawptr_aggregate

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

/// Reads the first two bytes of a slice and packs them little-endian. Taking
/// `&[u8]` (not an index) forces the subslice `&x0[1..]` at the call site to
/// materialize a fresh `&[u8]` fat pointer (the RawPtr aggregate).
#[device]
pub fn doot(s: &[u8]) -> u32 {
    (s[0] as u32) | ((s[1] as u32) << 8)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// `doot(&x0[1..])` — the `&x0[1..]` subslice lowers to a RawPtr aggregate.
    #[kernel]
    pub fn subslice(x0: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = doot(&x0[1..]);
        }
    }
}

fn main() {
    println!("=== slice_rawptr_aggregate: `doot(&x0[1..])` subslice on the GPU ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/slice_rawptr_aggregate.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    // x0 = [0x11, 0x22, 0x33, 0x44, 0x55]
    // &x0[1..] = [0x22, 0x33, 0x44, 0x55]
    // doot reads bytes [0x22, 0x33] => 0x22 | (0x33 << 8) = 0x3322 = 13090.
    let host_in: [u8; 5] = [0x11, 0x22, 0x33, 0x44, 0x55];
    let expected: u32 = (host_in[1] as u32) | ((host_in[2] as u32) << 8);

    let x0 = DeviceBuffer::from_host(&stream, &host_in).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    module
        .subslice(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(1),
            &x0,
            &mut out,
        )
        .expect("Kernel launch failed");

    let result = out.to_host_vec(&stream).unwrap()[0];

    if result == expected {
        println!(
            "slice_rawptr_aggregate: PASS (doot(&x0[1..]) = {result} = 0x{result:04X}, expected 0x{expected:04X})"
        );
    } else {
        println!(
            "slice_rawptr_aggregate: FAILED (got {result} = 0x{result:04X}, expected {expected} = 0x{expected:04X})"
        );
        std::process::exit(1);
    }
}
