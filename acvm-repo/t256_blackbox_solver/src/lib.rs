#![cfg_attr(not(test), warn(unused_crate_dependencies, unused_extern_crates))]

mod poseidon2;
mod poseidon2_constants;

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use ark_ff::{BigInt, MontConfig, Zero};
use ark_tom256::Affine;
use acir::{AcirField, BlackBoxFunc};
use acvm_blackbox_solver::{BlackBoxFunctionSolver, BlackBoxResolutionError};

type FieldElement = acir::acir_field::GenericFieldElement<ark_tom256::Fq>;

#[derive(Default)]
pub struct T256BlackboxSolver;

impl T256BlackboxSolver {
    fn coordinates_to_projective(
        x: FieldElement,
        y: FieldElement,
        is_infinite: FieldElement
    ) -> Result<ark_tom256::Projective, BlackBoxResolutionError> {
        Ok(
            if is_infinite.is_one() {
                ark_tom256::Projective::zero()
            } else {
                let p1 = Affine::new_unchecked(x.into_repr(), y.into_repr());
                if !p1.is_on_curve() {
                    return Err(BlackBoxResolutionError::Failed(
                        BlackBoxFunc::EmbeddedCurveAdd,
                        format!(
                            "Point ({}, {}) is not on curve",
                            x.to_hex(),
                            y.to_hex()
                        ),
                    ));
                }
                if !p1.is_in_correct_subgroup_assuming_on_curve() {
                    return Err(BlackBoxResolutionError::Failed(
                        BlackBoxFunc::EmbeddedCurveAdd,
                        format!(
                            "Point ({}, {}) is not in correct subgroup",
                            x.to_hex(),
                            y.to_hex()
                        ),
                    ));
                }
                ark_tom256::Projective::from(p1)
            }
        )
    }

    // Taken from the embedded_curve_ops in the Bn254 blackbox solver
    fn parse_msm_inputs(
        points: &[FieldElement],
        scalars_lo: &[FieldElement],
        scalars_hi: &[FieldElement],
    ) -> Result<(Vec<Affine>, Vec<BigInt<4>>), BlackBoxResolutionError>{
        if points.len() != 3 * scalars_lo.len() || scalars_lo.len() != scalars_hi.len() {
            return Err(BlackBoxResolutionError::Failed(
                BlackBoxFunc::MultiScalarMul,
                "Points and scalars must have the same length".to_string(),
            ));
        }

        // Collect all bases (affine points) and scalars for batch MSM
        let mut bases = Vec::new();
        let mut big_ints = Vec::new();

        for i in (0..points.len()).step_by(3) {
            if points[i + 2] > FieldElement::one() {
                return Err(BlackBoxResolutionError::Failed(
                    BlackBoxFunc::MultiScalarMul,
                    "EmbeddedCurvePoint is malformed (non-boolean `is_infinite` flag)".to_string(),
                ));
            }
            let point =
                Self::coordinates_to_projective(points[i], points[i + 1], points[i + 2])
                    .map_err(|e| BlackBoxResolutionError::Failed(BlackBoxFunc::MultiScalarMul, e.to_string()))?
                    .into_affine();

            let scalar_low: u128 =
                T256BlackboxSolver::field_to_u128_limb(&scalars_lo[i / 3], BlackBoxFunc::MultiScalarMul)?;

            let scalar_high: u128 =
                T256BlackboxSolver::field_to_u128_limb(&scalars_hi[i / 3], BlackBoxFunc::MultiScalarMul)?;

            // Convert to BigInt<4>, using u64 limbs.
            let limbs_array = [
                scalar_low as u64,
                (scalar_low >> 64) as u64,
                scalar_high as u64,
                (scalar_high >> 64) as u64,
            ];
            let scalar_bigint = BigInt::new(limbs_array);

            // Check if this is smaller than the grumpkin modulus
            if scalar_bigint >= ark_tom256::FrConfig::MODULUS {
                // Format as hex string (big-endian, most significant limb first)
                let hex_str = format!(
                    "{:016x}{:016x}{:016x}{:016x}",
                    limbs_array[3], limbs_array[2], limbs_array[1], limbs_array[0]
                );
                return Err(BlackBoxResolutionError::Failed(
                    BlackBoxFunc::MultiScalarMul,
                    format!("{hex_str} is not a valid T256 scalar"),
                ));
            }

            bases.push(point);
            big_ints.push(scalar_bigint);
        }
        Ok((bases, big_ints))
    }

    fn field_to_u128_limb(
        limb: &FieldElement,
        func: BlackBoxFunc,
    ) -> Result<u128, BlackBoxResolutionError> {
        limb.try_into_u128().ok_or_else(|| {
            BlackBoxResolutionError::Failed(
                func,
                format!("Limb {} is not less than 2^128", limb.to_hex()),
            )
        })
    }
}

impl BlackBoxFunctionSolver<FieldElement> for T256BlackboxSolver {
    fn multi_scalar_mul(
        &self,
        points: &[FieldElement],
        scalars_lo: &[FieldElement],
        scalars_hi: &[FieldElement],
        predicate: bool
    ) -> Result<(FieldElement, FieldElement, FieldElement), BlackBoxResolutionError> {
        if !predicate {
            return Ok(
                (
                    FieldElement::zero(),
                    FieldElement::zero(),
                    FieldElement::one(),
                )
            );
        }

        let (points, scalars) = Self::parse_msm_inputs(points, scalars_lo, scalars_hi)?;

        let msm_result = Affine::from(
            ark_tom256::Projective::msm_bigint(&points, &scalars)
        );

        if let Some((out_x, out_y)) = msm_result.xy() {
            Ok((
                FieldElement::from_repr(out_x),
                FieldElement::from_repr(out_y),
                FieldElement::from(u128::from(msm_result.is_zero())),
            ))
        } else {
            Ok((FieldElement::zero(), FieldElement::zero(), FieldElement::one()))
        }
    }

    fn ec_add(
        &self,
        input1_x: &FieldElement,
        input1_y: &FieldElement,
        input1_infinite: &FieldElement,
        input2_x: &FieldElement,
        input2_y: &FieldElement,
        input2_infinite: &FieldElement,
        predicate: bool
    ) -> Result<(FieldElement, FieldElement, FieldElement), BlackBoxResolutionError> {
        if !predicate {
            return Ok(
                (
                    FieldElement::zero(),
                    FieldElement::zero(),
                    FieldElement::one(),
                )
            );
        }

        let p1 = T256BlackboxSolver::coordinates_to_projective(*input1_x, *input1_y, *input1_infinite)?;
        let p2 = T256BlackboxSolver::coordinates_to_projective(*input2_x, *input2_y, *input2_infinite)?;

        let sum = Affine::from(p1 + p2);
        if let Some((x, y)) = sum.xy() {
            Ok((FieldElement::from_repr(x), FieldElement::from_repr(y), FieldElement::zero()))
        } else {
            Ok((FieldElement::zero(), FieldElement::zero(), FieldElement::one()))
        }
    }

    fn poseidon2_permutation(&self, inputs: &[FieldElement]) -> Result<Vec<FieldElement>, BlackBoxResolutionError> {
        Ok(
            // TODO: this is obviously wrong and just here for the purpose of the comptime code from
            //     noir stdlib to compile
            inputs.to_vec()
            // poseidon2::poseidon2_permutation(inputs)?
        )
    }
}