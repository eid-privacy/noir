<div align="center">
  <picture>
    <img src="./noir-logo.png" alt="The Noir Programming Language" width="35%">
  </picture>

[Website][Noir] | [Getting started] | [Documentation] | [Contributing]
</div>



> [!NOTE]
> **This is the [eid-privacy](https://github.com/eid-privacy) fork of Noir, `tom-256` branch.**
> It was developed in the Innosuisse project on privacy-preserving e-ID
> ([blog](https://eid-privacy.github.io)) and is used by
> [spartan-backend](https://github.com/eid-privacy/spartan-backend).
> See [Tom-256 support](#tom-256-support-eid-privacy-fork) below.
> The `master` branch tracks upstream [noir-lang/noir](https://github.com/noir-lang/noir).

# The Noir Programming Language

[![Non-deterministic fuzz tests](https://github.com/noir-lang/noir/actions/workflows/nightly-fuzz-test.yml/badge.svg)](https://github.com/noir-lang/noir/actions/workflows/nightly-fuzz-test.yml)

Noir is a Domain Specific Language for SNARK proving systems. It has been designed to use any ACIR compatible proving system.

**This implementation is in early development. It has not been reviewed or audited. It is not suitable to be used in production. Expect bugs!**

## Tom-256 support (eid-privacy fork)

This branch adds the [Tom-256](https://eprint.iacr.org/2021/1183) curve as the native field of Noir,
instead of BN254. The order of Tom-256 equals the base field of P-256 (secp256r1). So P-256 point
arithmetic, e.g. for ECDSA signature verification of e-ID credentials, can be done natively in the
circuit without emulating non-native field arithmetic. The proofs are generated with a Spartan prover
([vega-prover](https://github.com/microsoft/vega-prover)) through
[spartan-backend](https://github.com/eid-privacy/spartan-backend).

Changes compared to upstream Noir `v1.0.0-beta.26`:

- `t256` cargo feature on the compiler and ACVM crates, enabled by default in `nargo` on this branch.
  The resulting binary is often installed as `nargo-t256`.
- New `acvm-repo/t256_blackbox_solver`: blackbox solver for Tom-256, including embedded curve
  operations (P-256 points) and Poseidon2 over the Tom-256 field.
- `noir_stdlib`: embedded curve operations on P-256 points over the Tom-256 field (`field/t256.nr`).
- Fixes from the [zkSecurity review](https://github.com/eid-privacy/zkp-reports/tree/main/security-review)
  (curve membership checks, scalar handling in MSM).

Releases are tagged `t256-v0.<noir beta>[-<n>]`, e.g. `t256-v0.26-2` (Noir `v1.0.0-beta.26`).
The [eid-privacy/flakes](https://github.com/eid-privacy/flakes) repository has Nix packages for
these tags.

Related repositories:

- [spartan-backend](https://github.com/eid-privacy/spartan-backend): Spartan prover for ACIR circuits
  compiled with this fork
- [ark-tom256](https://github.com/eid-privacy/ark-tom256): arkworks implementation of the Tom-256 curve
- [flakes](https://github.com/eid-privacy/flakes): Nix flake providing `nargo-t256`

**This fork is a research prototype and is not suitable for production use.**

## Quick Start

Read the [installation section][Getting started] from the [Noir docs][Documentation].

Once you have read through the documentation, you can visit [Awesome Noir](https://github.com/noir-lang/awesome-noir) to run some of the examples that others have created.

## Getting Help

Join the Noir [forum][Forum] or [Discord][Discord]

## Contributing

See [CONTRIBUTING.md][CONTRIBUTING].

## Future Work

The current focus is to gather as much feedback as possible while in the alpha phase. The main focuses of Noir are _safety_ and _developer experience_. If you find a feature that does not seem to be in line with these goals, please open an issue!

## Minimum Rust version

This workspace's minimum supported rustc version is 1.89.0.

## License

Noir is free and open source. It is distributed under a dual license. (MIT/APACHE)

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this repository by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

[Noir]: https://www.noir-lang.org/
[Getting Started]: https://noir-lang.org/docs/getting_started/quick_start/
[Forum]: https://forum.aztec.network/c/noir
[Discord]: https://discord.gg/JtqzkdeQ6G
[Documentation]: https://noir-lang.org/docs/
[Contributing]: CONTRIBUTING.md
