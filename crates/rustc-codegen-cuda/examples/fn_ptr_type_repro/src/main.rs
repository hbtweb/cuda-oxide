/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for fn-item / fn-pointer **type translation**.
//!
//! `RigidTy::FnDef` (the zero-sized type of a named `fn`) and `RigidTy::FnPtr`
//! (`fn(T) -> U`) previously hit the "Type translation not yet implemented"
//! wildcard in `translator::types::translate_type`, so any kernel that even
//! *named* a fn pointer failed to import. The fix maps both to a generic
//! opaque pointer.
//!
//! This example exercises the TYPE path without an indirect call (PTX has
//! limited support for calling through a fn pointer; that lowering is a
//! separate follow-up). The kernel coerces a named `fn` to a
//! `fn(u32) -> u32` pointer and compares that pointer to ITSELF — so the
//! fn-pointer type must be translated, but no value is ever *called* through
//! the pointer.
//!
//! Scope of the fix this guards: fn-item / fn-pointer **type translation**
//! only. The minimal fix maps every `FnDef`/`FnPtr` to a generic opaque
//! pointer, which is enough to import, store, pass and reflexively compare a
//! fn pointer. It deliberately does NOT materialize distinct per-fn addresses,
//! so comparing two *different* fn pointers for inequality is out of scope
//! (it depends on fn-address value lowering, a separate follow-up). The test
//! therefore asserts only the reflexive `f == f` case, which the type fix
//! fully supports.
//!
//! Pre-fix this fails to import (unsupported FnDef/FnPtr type); post-fix it
//! imports, builds, and the host verifies the reflexive pointer comparison.
//!
//! Build and run with:
//!   cargo oxide run fn_ptr_type_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    fn inc(x: u32) -> u32 {
        x.wrapping_add(1)
    }

    /// Forces `FnDef` -> `FnPtr` type translation by binding a named `fn` to a
    /// `fn(u32) -> u32` local, then exercises the fn-pointer TYPE through a
    /// reflexive pointer comparison (no indirect call). Writes:
    ///   out[0] = (f == f) as u32   -> 1
    #[kernel]
    pub fn fn_ptr_eq(mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() == 0 {
            let f: fn(u32) -> u32 = inc;

            // Reflexive `fn`-pointer equality. This needs the fn-pointer TYPE
            // to be translatable, but never calls through the pointer and does
            // not depend on distinct per-fn addresses.
            let same = core::ptr::fn_addr_eq(f, f) as u32;

            unsafe {
                *out.get_unchecked_mut(0) = same;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== fn_ptr_type_repro: FnDef/FnPtr type translation on the GPU ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    const N: usize = 1;
    let cfg = LaunchConfig::for_num_elems(1);

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    module.fn_ptr_eq(stream.as_ref(), cfg, &mut out)?;
    let got = out.to_host_vec(&stream)?;

    let mut failed = 0u32;
    // The kernel imported, built and launched at all — meaning the fn-pointer
    // TYPE was translated (pre-fix it hit the "not yet implemented" wildcard).
    // The reflexive comparison must hold.
    if got[0] == 1 {
        println!("PASS fn_addr_eq(inc, inc) == true (fn-pointer type translated)");
    } else {
        println!("FAIL fn_addr_eq(inc, inc): got {} expected 1", got[0]);
        failed += 1;
    }

    if failed != 0 {
        println!("\nfn_ptr_type_repro: FAILED ({failed} mismatches)");
        std::process::exit(1);
    }

    println!("\n✓ SUCCESS: fn_ptr_type_repro (FnDef/FnPtr type translation)");
    Ok(())
}
