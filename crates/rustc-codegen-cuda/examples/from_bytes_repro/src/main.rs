/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression guard for the `[N x i8]` → `iN` aggregate→scalar bitcast fix.
//!
//! `u32::from_ne_bytes([u8; 4])` / `u64::from_le_bytes([u8; 8])` lower in MIR
//! to a `Transmute` from an LLVM `[N x i8]` array to an `iN` integer. The cast
//! lowering had no arm for an `ArrayType` source, so it fell through to the
//! trailing `else` and emitted an illegal `bitcast [N x i8] to iN` — LLVM
//! forbids bitcasting between aggregates and scalars, so device codegen
//! failed module verification (the issue-#125 class).
//!
//! The fix adds an `ArrayType → IntegerType` arm that round-trips through
//! memory (alloca + store + load), the same shape the struct → struct arm
//! already uses. This example builds two such transmutes on the device and the
//! host verifies the reconstructed integers.
//!
//! Pre-fix this fails device codegen; post-fix it builds and verifies.
//!
//! Build and run with:
//!   cargo oxide run from_bytes_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// `u32::from_ne_bytes` → `[u8; 4]` to `i32`/`u32` round-trip via memory.
    /// Bytes are little-endian on NVPTX, so `[4,0,0,0]` reconstructs to 4.
    #[kernel]
    pub fn from_ne_bytes_u32(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let a = u32::from_ne_bytes([4, 0, 0, 0]);
            let b = u32::from_le_bytes([0, 1, 0, 0]); // 256
            *slot = a.wrapping_add(b.wrapping_mul(1_000_000));
        }
    }

    /// `u64::from_le_bytes` → `[u8; 8]` to `i64`/`u64` round-trip via memory.
    /// `[1,0,0,0,0,0,0,0]` reconstructs to 1.
    #[kernel]
    pub fn from_le_bytes_u64(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let a = u64::from_le_bytes([1, 0, 0, 0, 0, 0, 0, 0]);
            let b = u64::from_ne_bytes([0, 0, 0, 0, 2, 0, 0, 0]); // 2 << 32
            *slot = a.wrapping_add(b);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== from_bytes_repro: u32/u64::from_*_bytes ([N x i8] -> iN) on the GPU ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    const N: usize = 1;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let mut failed = 0u32;

    // ---- u32 ----
    {
        let mut out = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.from_ne_bytes_u32(stream.as_ref(), cfg, &mut out)?;
        let got = out.to_host_vec(&stream)?[0];
        // a = 4, b = 256 -> 4 + 256 * 1_000_000
        let expected = 4u32.wrapping_add(256u32.wrapping_mul(1_000_000));
        if got == expected {
            println!("PASS u32::from_ne_bytes/from_le_bytes: {got}");
        } else {
            println!("FAIL u32: got {got} expected {expected}");
            failed += 1;
        }
    }

    // ---- u64 ----
    {
        let mut out = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        module.from_le_bytes_u64(stream.as_ref(), cfg, &mut out)?;
        let got = out.to_host_vec(&stream)?[0];
        // a = 1, b = 2 << 32 -> 1 + (2 << 32)
        let expected = 1u64.wrapping_add(2u64 << 32);
        if got == expected {
            println!("PASS u64::from_le_bytes/from_ne_bytes: {got}");
        } else {
            println!("FAIL u64: got {got} expected {expected}");
            failed += 1;
        }
    }

    if failed != 0 {
        println!("\nfrom_bytes_repro: FAILED ({failed} mismatches)");
        std::process::exit(1);
    }

    println!("\n✓ SUCCESS: from_bytes_repro");
    Ok(())
}
