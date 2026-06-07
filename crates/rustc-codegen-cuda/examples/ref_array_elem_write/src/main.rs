/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for `&mut (*arr)[i]` writes through a referenced array
//! element not being persisted (silent dropped write).
//!
//! Taking `&mut arr[i]` where `arr: &mut [f32; N]` lowers to
//! `Rvalue::Ref(place = [Deref, Index])`. The importer's address helper
//! `translate_place_addr_from_slot` punted on a leading `Deref` (catch-all
//! `Ok(None)`), so the `Rvalue::Ref` arm fell through to its `MirRefOp`
//! fallback, which materializes the element as a LOADED value and stores that
//! COPY in a fresh stack slot. The subsequent `*e = 42.0` then wrote the copy,
//! not the array — a silent miscompile (the array element stays 0).
//!
//! The fix teaches `translate_place_addr_from_slot` to address through `Deref`
//! (load the inner pointer, keeping the result a pointer) and runtime `Index`
//! (emit `MirArrayElementAddrOp` into the array pointee), so the write lands in
//! the real array.
//!
//! Pre-fix the written elements stay 0; post-fix they are 42.0.
//!
//! Build and run with:
//!   cargo oxide run ref_array_elem_write

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

const N: usize = 8;

#[cuda_module]
mod kernels {
    use super::*;

    /// Write `42.0` through a `&mut` to each element of a fixed-size array,
    /// addressed via `&mut arr[i]` where `arr: &mut [f32; N]` (the
    /// `[Deref, Index]` place the fix targets), then copy the array to `out`.
    ///
    /// `out` is `&[f32]` aliasing a writable device buffer; thread 0 does all
    /// the work and writes `N` elements through a raw pointer (the raw-write
    /// trick the sibling `struct_field_reorder` example uses), because a
    /// `DisjointSlice` is one-element-per-thread.
    #[kernel]
    pub fn ref_array_elem_write(out: &[f32]) {
        let tid = thread::index_1d();
        if tid.get() != 0 {
            return;
        }

        let mut data: [f32; N] = [0.0; N];
        // `&mut [f32; N]` reference local — taking `&mut arr[i]` off THIS gives
        // the `[Deref, Index]` projection (the fix's target shape). A bare
        // local array would give just `[Index]`.
        let arr: &mut [f32; N] = &mut data;

        let mut i = 0usize;
        while i < N {
            let e: &mut f32 = &mut arr[i];
            *e = 42.0;
            i += 1;
        }

        unsafe {
            let p = out.as_ptr() as *mut f32;
            let mut j = 0usize;
            while j < N {
                *p.add(j) = data[j];
                j += 1;
            }
        }
    }
}

fn main() {
    println!("=== ref_array_elem_write: &mut (*arr)[i] write through a referenced element ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/ref_array_elem_write.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    let out = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    module
        .ref_array_elem_write(stream.as_ref(), LaunchConfig::for_num_elems(1), &out)
        .expect("Kernel launch failed");

    let result = out.to_host_vec(&stream).unwrap();

    let mut errors = 0usize;
    for (j, &v) in result.iter().enumerate() {
        if v != 42.0 {
            if errors < 8 {
                eprintln!("  out[{j}]: got {v}, expected 42.0");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!(
            "ref_array_elem_write: PASS (all {N} elements written through &mut arr[i] == 42.0)"
        );
    } else {
        println!("ref_array_elem_write: FAILED ({errors} elements not 42.0)");
        std::process::exit(1);
    }
}
