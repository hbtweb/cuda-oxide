/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for the `Option<&T>` reference-to-scalar constant import fix.
//!
//! With a `Some`/`None` pair of `Option<&u32>` live in one frame, rustc -O
//! const-folds `None.unwrap_or(&77)` to a reference-to-SCALAR constant `&77` —
//! a `ConstantKind::Allocated` whose 8 pointer bytes are a relocation
//! placeholder (zero), with the real target (the `77` allocation) recorded in
//! `alloc.provenance.ptrs`.
//!
//! The MIR importer's pointer-constant arm followed provenance only when the
//! pointee was a STRUCT (via `translate_struct_constant` + `mir.ref`). For a
//! scalar pointee it skipped that and fell through to the raw-pointer branch,
//! which emitted `inttoptr i64 0` — dropping the relocation. The later
//! `*b.unwrap_or(&77)` then did `load i32, ptr null` => CUDA 700 (illegal
//! address) or garbage.
//!
//! The fix broadens the provenance gate to "has provenance" and, for a scalar
//! pointee, follows the provenance target allocation's bytes, materializes the
//! pointee value via `translate_constant_value_from_bytes`, and wraps it in a
//! `MirRefOp` — exactly the struct path, but for a scalar.
//!
//! Pre-fix `out[1]` is a null-deref (CUDA 700 / garbage). Post-fix the kernel
//! runs and the host verifies `out[0] == 5` (the `Some(&r)` value) and
//! `out[1] == 77` (the `None.unwrap_or(&77)` default).
//!
//! Build and run with:
//!   cargo oxide run option_ref_unwrap_or

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 builds a `Some(&r)` and a `None` of type `Option<&u32>`, then
    /// writes `*a.unwrap_or(&77)` to `out[0]` and `*b.unwrap_or(&77)` to
    /// `out[1]`. The `&77` default is the reference-to-scalar constant under
    /// test. Keeping BOTH a `Some` and a `None` in the same frame is what makes
    /// rustc const-fold the `None` branch's `unwrap_or` to the `&77` constant.
    ///
    /// `out` is `&[u32]` aliasing a writable device buffer; thread 0 writes two
    /// elements through a raw pointer (the same raw-write trick the sibling
    /// `struct_field_reorder` example uses) because a `DisjointSlice` is
    /// one-element-per-thread.
    #[kernel]
    pub fn opt_ref_unwrap_or(out: &[u32]) {
        let tid = thread::index_1d();
        if tid.get() != 0 {
            return;
        }
        let r: u32 = 5;
        let a: Option<&u32> = Some(&r);
        let b: Option<&u32> = None;

        let v0: u32 = *a.unwrap_or(&77);
        let v1: u32 = *b.unwrap_or(&77);

        unsafe {
            let p = out.as_ptr() as *mut u32;
            *p.add(0) = v0;
            *p.add(1) = v1;
        }
    }
}

fn main() {
    println!("=== option_ref_unwrap_or: None.unwrap_or(&77) reference-to-scalar constant ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/option_ref_unwrap_or.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    let out = DeviceBuffer::<u32>::zeroed(&stream, 2).unwrap();

    module
        .opt_ref_unwrap_or(stream.as_ref(), LaunchConfig::for_num_elems(1), &out)
        .expect("Kernel launch failed");

    let result = out.to_host_vec(&stream).unwrap();

    // out[0] = *Some(&r).unwrap_or(&77)  = *(&5)  = 5
    // out[1] = *None.unwrap_or(&77)      = *(&77) = 77
    let want = [5u32, 77u32];

    if result[0] == want[0] && result[1] == want[1] {
        println!(
            "option_ref_unwrap_or: PASS (out[0]={} expected 5, out[1]={} expected 77)",
            result[0], result[1]
        );
    } else {
        println!(
            "option_ref_unwrap_or: FAILED (out[0]={} expected 5, out[1]={} expected 77)",
            result[0], result[1]
        );
        std::process::exit(1);
    }
}
