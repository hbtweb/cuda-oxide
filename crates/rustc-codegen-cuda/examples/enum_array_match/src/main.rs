/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for the enum-in-array `match` projection fix.
//!
//! `match xs[i] { E::A(x) => ..., E::B(y) => ..., E::C => ... }` lowers to a
//! MIR projection chain `[Index, Downcast, Field]` (or `[ConstantIndex,
//! Downcast, Field]` for a literal index). The importer's iterative place
//! walker (`translate_place_iterative`) advanced the SSA *value* in the
//! `Index` / `ConstantIndex` arms but never narrowed the running *Rust type*,
//! which stayed at the outer `Array(Adt(E), N)`. The subsequent `Downcast` /
//! `Field` step then handed that array type to the enum-payload projection,
//! which rejected it with `Downcast on non-ADT type: Array`.
//!
//! The fix narrows the running Rust type to the array/slice element type in
//! both index arms, so the Downcast/Field step sees `Adt(E)`.
//!
//! Pre-fix this FAILS import with `Downcast on non-ADT type: Array`.
//! Post-fix it builds and the host verifies the match results.
//!
//! Build and run with:
//!   cargo oxide run enum_array_match

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// An enum with payload-carrying and unit variants, stored in a device-local
/// array and matched on. `Copy` so it can sit in a `[E; N]` literal.
#[derive(Clone, Copy)]
pub enum E {
    A(u32),
    B(u32),
    C,
}

#[cuda_module]
mod kernels {
    use super::*;

    /// `match xs[i]` with a RUNTIME index — exercises the `Index` arm of the
    /// iterative place walker (projection chain `[Index, Downcast, Field]`).
    #[kernel]
    pub fn match_runtime_index(index: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 4] = [E::A(7), E::B(8), E::C, E::A(100)];
            let i = index as usize;
            let r = match xs[i] {
                E::A(x) => x,
                E::B(y) => y + 1000,
                E::C => 9999,
            };
            *out_elem = r;
        }
    }

    /// `match xs[0]` with a CONSTANT index — exercises the `ConstantIndex` arm
    /// (projection chain `[ConstantIndex, Downcast, Field]`).
    #[kernel]
    pub fn match_const_index(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let xs: [E; 4] = [E::A(7), E::B(8), E::C, E::A(100)];
            let r = match xs[0] {
                E::A(x) => x,
                E::B(y) => y + 1000,
                E::C => 9999,
            };
            *out_elem = r; // xs[0] == E::A(7) => 7
        }
    }
}

fn run_runtime(
    module: &kernels::LoadedModule,
    stream: &Arc<CudaStream>,
    index: u32,
    expected: u32,
) -> bool {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig::for_num_elems(1);
    module
        .match_runtime_index((stream).as_ref(), config, index, &mut d_out)
        .expect("Kernel launch failed");
    let result = d_out.to_host_vec(stream).unwrap()[0];
    if result == expected {
        println!("  match xs[{index}]: PASS (result = {result})");
        true
    } else {
        println!("  match xs[{index}]: FAIL (expected {expected}, got {result})");
        false
    }
}

fn run_const(module: &kernels::LoadedModule, stream: &Arc<CudaStream>, expected: u32) -> bool {
    let mut d_out = DeviceBuffer::<u32>::zeroed(stream, 1).unwrap();
    let config = LaunchConfig::for_num_elems(1);
    module
        .match_const_index((stream).as_ref(), config, &mut d_out)
        .expect("Kernel launch failed");
    let result = d_out.to_host_vec(stream).unwrap()[0];
    if result == expected {
        println!("  match xs[0] (const): PASS (result = {result})");
        true
    } else {
        println!("  match xs[0] (const): FAIL (expected {expected}, got {result})");
        false
    }
}

fn main() {
    println!("=== enum_array_match: `match xs[i]` over an enum array on the GPU ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/enum_array_match.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    // xs = [E::A(7), E::B(8), E::C, E::A(100)]
    //   match: A(x)=>x, B(y)=>y+1000, C=>9999
    //   => [7, 1008, 9999, 100]
    let mut ok = true;
    ok &= run_runtime(&module, &stream, 0, 7);
    ok &= run_runtime(&module, &stream, 1, 1008);
    ok &= run_runtime(&module, &stream, 2, 9999);
    ok &= run_runtime(&module, &stream, 3, 100);
    ok &= run_const(&module, &stream, 7);

    if ok {
        println!("\nenum_array_match: PASS (all match arms over the enum array verified)");
    } else {
        println!("\nenum_array_match: FAILED");
        std::process::exit(1);
    }
}
