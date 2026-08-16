// SPDX-License-Identifier: Apache-2.0

// Two lints are refused crate-wide, and both refusals are the bit-exactness contract
// rather than taste. See docs/RUST_PORT.md, "Translation rules".
//
// `needless_range_loop` wants `for (i, x) in v.iter().enumerate()` wherever the C wrote
// `for (i = 0; i < n; i++)`. In a kernel that is not a cosmetic change: the whole port is
// graded on reproducing the C reduction order bit for bit, and RUST_PORT.md:248 states
// that "compaction that hides the reduction order is a regression even when it compiles
// and passes." An indexed loop over an explicit bound is the form that stays diffable
// against src/core/k3_ops.c, which is the property the review process depends on.
//
// `too_many_arguments` fires on the transliterated kernel signatures. k3_ops.c passes
// dims and weight pointers positionally; bundling them into structs to satisfy a
// threshold of seven would break the same line-for-line correspondence.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

pub mod accel;
pub mod arch;
pub mod bind;
pub mod cache;
pub mod capability;
pub mod certsparse;
pub mod chat;
pub mod cfg;
pub mod downt;
pub mod dataset;
pub mod dspark;
pub mod fetch;
pub mod fmt;
pub mod gqa;
pub mod gguf;
pub mod io;
pub mod libm;
pub mod lut;
pub mod model;
pub mod moearch;
pub mod moegen;
pub mod ops;
pub mod prefix;
pub mod qwen35;
pub mod qwen35run;
pub mod registry;
pub mod sample;
pub mod serve;
pub mod st;
pub mod tok;
pub mod tok_k3;
pub mod train;
pub mod v4;
pub mod v4run;
pub mod vram;
