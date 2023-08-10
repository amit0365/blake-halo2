use crate::blake2f::table16::gate::Gate;

use super::super::util::*;

use group::ff::{Field, PrimeField};
use halo2_proofs::plonk::{Constraint, Constraints, Expression};
use std::marker::PhantomData;

pub struct MixingGate<F: Field>(PhantomData<F>);

impl<F: PrimeField> MixingGate<F> {
    fn ones() -> Expression<F> {
        Expression::Constant(F::ONE)
    }

    // Decompose `A,B,C,D` words
    // (16, 16, 16, 16)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_abcd(
        s_decompose_abcd: Expression<F>,
        a: Expression<F>,
        spread_a: Expression<F>,
        b: Expression<F>,
        spread_b: Expression<F>,
        c: Expression<F>,
        spread_c: Expression<F>,
        d: Expression<F>,
        spread_d: Expression<F>,
        word_lo: Expression<F>,
        spread_word_lo: Expression<F>,
        word_mo: Expression<F>,
        spread_word_mo: Expression<F>,
        word_el: Expression<F>,
        spread_word_el: Expression<F>,
        word_hi: Expression<F>,
        spread_word_hi: Expression<F>,
    ) -> Constraints<
        F,
        (&'static str, Expression<F>),
        impl Iterator<Item = (&'static str, Expression<F>)>,
    > {
        let dense_check = a
            + b * F::from(1 << 16)
            + c * F::from(1 << 32)
            + d * F::from(1 << 48)

            + word_lo * (-F::ONE)
            + word_mo * F::from(1 << 16) * (-F::ONE)
            + word_el * F::from(1 << 32) * (-F::ONE)
            + word_hi * F::from(1 << 48) * (-F::ONE);

        let spread_check = spread_a
            + spread_b * F::from(1 << 32)
            + spread_c * F::from(1 << 64)
            + spread_d * F::from(1 << 96)

            + spread_word_lo * (-F::ONE)
            + spread_word_mo * F::from(1 << 32) * (-F::ONE)
            + spread_word_el * F::from(1 << 64) * (-F::ONE)
            + spread_word_hi * F::from(1 << 96) * (-F::ONE);

        Constraints::with_selector(
            s_decompose_abcd,
            dense_check
                .chain(Some(("spread_check", spread_check)))
        )

    }


// Decompose `H,G,F,E` words
    // (16, 16, 8, 8, 16)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_efgh(
        s_decompose_efgh: Expression<F>,
        e: Expression<F>,
        spread_e: Expression<F>,
        tag_f_lo: Expression<F>,
        f_lo: Expression<F>,
        spread_f_lo: Expression<F>,
        tag_f_hi: Expression<F>,
        f_hi: Expression<F>,
        spread_f_hi: Expression<F>,
        g: Expression<F>,
        spread_g: Expression<F>,
        h: Expression<F>,
        spread_h: Expression<F>,
        word_lo: Expression<F>,
        spread_word_lo: Expression<F>,
        word_mo: Expression<F>,
        spread_word_mo: Expression<F>,
        word_el: Expression<F>,
        spread_word_el: Expression<F>,
        word_hi: Expression<F>,
        spread_word_hi: Expression<F>,
    ) -> Constraints<
        F,
        (&'static str, Expression<F>),
        impl Iterator<Item = (&'static str, Expression<F>)>,
    > {
        // bits are range checked while looking up spread
        let dense_check = e
            + f_lo * F::from(1 << 8)
            + f_hi * F::from(1 << 16)
            + g * F::from(1 << 32)
            + h * F::from(1 << 48)

            + word_lo * (-F::ONE)
            + word_mo * F::from(1 << 16) * (-F::ONE)
            + word_el * F::from(1 << 32) * (-F::ONE)
            + word_hi * F::from(1 << 48) * (-F::ONE);

        let range_check_tag_f_lo: Expression<F> = Gate::range_check(tag_f_lo, 0, 0);
        let range_check_tag_f_hi: Expression<F> = Gate::range_check(tag_f_hi, 0, 0);

        let spread_check = spread_e
            + spread_f_lo * F::from(1 << 16)
            + spread_f_hi * F::from(1 << 32)
            + spread_g * F::from(1 << 64)
            + spread_h * F::from(1 << 96)

            + spread_word_lo * (-F::ONE)
            + spread_word_mo * F::from(1 << 32) * (-F::ONE)
            + spread_word_el * F::from(1 << 64) * (-F::ONE)
            + spread_word_hi * F::from(1 << 96) * (-F::ONE);

        Constraints::with_selector(
            s_decompose_efgh,
            dense_check
                .chain(Some(("spread_check", spread_check)))
                .chain(Some(("range_check_tag_f_lo", range_check_tag_f_lo)))
                .chain(Some(("range_check_tag_f_hi", range_check_tag_f_hi))),
        )

    }



    // Decompose `IJKL` words
    // (1, 15, 16, 16, 16)-bit chunks
    #[allow(clippy::too_many_arguments)]
    pub fn s_decompose_ijkl(
        s_decompose_efgh: Expression<F>,
        i_lo: Expression<F>,
        spread_i_lo: Expression<F>,
        tag_i_hi: Expression<F>,
        i_hi: Expression<F>,
        spread_i_hi: Expression<F>,
        j: Expression<F>,
        spread_j: Expression<F>,
        k: Expression<F>,
        spread_k: Expression<F>,
        l: Expression<F>,
        spread_l: Expression<F>,
        word_lo: Expression<F>,
        spread_word_lo: Expression<F>,
        word_mo: Expression<F>,
        spread_word_mo: Expression<F>,
        word_el: Expression<F>,
        spread_word_el: Expression<F>,
        word_hi: Expression<F>,
        spread_word_hi: Expression<F>,
    ) -> Constraints<
        F,
        (&'static str, Expression<F>),
        impl Iterator<Item = (&'static str, Expression<F>)>,
    > { 
        let dense_check = i_lo
            + i_hi * F::from(1 << 1)
            + j * F::from(1 << 16)
            + k * F::from(1 << 32)
            + l * F::from(1 << 48)

            + word_lo * (-F::ONE)
            + word_mo * F::from(1 << 16) * (-F::ONE)
            + word_el * F::from(1 << 32) * (-F::ONE)
            + word_hi * F::from(1 << 48) * (-F::ONE);

        let spread_check = spread_i_lo
            + spread_i_hi * F::from(1 << 2)
            + spread_j * F::from(1 << 32)
            + spread_k * F::from(1 << 64)
            + spread_l * F::from(1 << 96)

            + spread_word_lo * (-F::ONE)
            + spread_word_mo * F::from(1 << 32) * (-F::ONE)
            + spread_word_el * F::from(1 << 64) * (-F::ONE)
            + spread_word_hi * F::from(1 << 96) * (-F::ONE);

        let range_check_tag_i_hi: Expression<F> = Gate::range_check(tag_i_hi, 0, 1);

        Constraints::with_selector(
            s_decompose_efgh,
            dense_check
                .chain(Some(("spread_check", spread_check)))
                .chain(Some(("range_check_tag_i_hi", range_check_tag_i_hi))),
        )

    }


// First gate addition modulo, Va ← Va + Vb + x  with input
// todo change decomposition of words a,b,c,d after each rounds??
// only one carry since carries for each splits are propagated into next split 
#[allow(clippy::too_many_arguments)]
    pub fn s_vector_a1(
        s_vector_a1: Expression<F>,
        vector_m_a1: Expression<F>,
        vector_n_a1: Expression<F>,
        vector_o_a1: Expression<F>,
        vector_p_a1: Expression<F>,
        vector_a1_carry: Expression<F>,
        vector_m_a: Expression<F>,
        vector_n_a: Expression<F>,
        vector_o_a: Expression<F>,
        vector_p_a: Expression<F>,
        vector_m_b: Expression<F>,
        vector_n_b: Expression<F>,
        vector_o_b: Expression<F>,
        vector_p_b: Expression<F>,        
        vector_m_x: Expression<F>,
        vector_n_x: Expression<F>,
        vector_o_x: Expression<F>,
        vector_p_x: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {

        let vector_a = vector_m_a + vector_n_a * F::from(1 << 32) + vector_o_a * F::from(1 << 64) + vector_p_a * F::from(1 << 96);
        let vector_b = vector_m_b + vector_n_b * F::from(1 << 32) + vector_o_b * F::from(1 << 64) + vector_p_b * F::from(1 << 96);
        let vector_x = vector_m_x + vector_n_x * F::from(1 << 32) + vector_o_x * F::from(1 << 64) + vector_p_x * F::from(1 << 96);
        let vector_sum = vector_a + vector_b + vector_x;

        let vector_a1 = vector_m_a1 + vector_n_a1 * F::from(1 << 16) + vector_o_a1 * F::from(1 << 32) + vector_p_a1 * F::from(1 << 48);

        // WITNESS ADDITION
        let check = vector_sum - (vector_a1_carry * F::from(1 << 32)) - vector_a1;

        Some(("vector_a1", s_vector_a1 * check))
    }


// Second gate, xor and bit rotations: Vd1 ← (Vd xor Va1) >>> 32
// vector_d1 on abcd words
    #[allow(clippy::too_many_arguments)]
    pub fn s_vector_d1(
        s_vector_d1: Expression<F>,
        spread_m_d1_even: Expression<F>,
        spread_m_d1_odd: Expression<F>,
        spread_n_d1_even: Expression<F>,
        spread_n_d1_odd: Expression<F>,
        spread_o_d1_even: Expression<F>,
        spread_o_d1_odd: Expression<F>,
        spread_p_d1_even: Expression<F>,
        spread_p_d1_odd: Expression<F>,
        spread_m_a1: Expression<F>,
        spread_n_a1: Expression<F>,
        spread_o_a1: Expression<F>,
        spread_p_a1: Expression<F>,
        spread_m_d: Expression<F>,
        spread_n_d: Expression<F>,
        spread_o_d: Expression<F>,
        spread_p_d: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {

        // witnessing that the addition result
        let spread_witness = 
               spread_m_d1_even + spread_m_d1_odd * F::from(2)
            + (spread_n_d1_even + spread_n_d1_odd * F::from(2)) * F::from(1 << 32)
            + (spread_o_d1_even + spread_o_d1_odd * F::from(2)) * F::from(1 << 64)
            + (spread_p_d1_even + spread_p_d1_odd * F::from(2)) * F::from(1 << 96);

        //     Vd1 ← (Vd xor Va1) >>> 32

        //     addition to get xor in spread form

        //      s_p_a1  |  s_o_a1  |  s_n_a1  |  s_m_a1  |
        //      s_p_d   |  s_o_d   |  s_n_d   |  s_m_d   |

        // after adding the chunks we do rotation
        // rotating 32 bits dense -> rotating two 32bit chunks in spread form

        //  s_n_a1 + s_n_d  |  s_m_a1 + s_m_d |  s_p_a1 + s_p_d  |  s_o_a1 + s_o_d  |

        let rot = spread_o_a1.clone() + spread_o_d.clone()
            + (spread_p_a1.clone() + spread_p_d.clone())* F::from(1 << 32)
            + (spread_m_a1.clone() + spread_m_d.clone())* F::from(1 << 64)
            + (spread_n_a1.clone() + spread_n_d.clone())* F::from(1 << 96);


        let check = spread_witness + (rot * -F::ONE);

        Some(("vector_d1", s_vector_d1 * check))

    }


    // Third gate addition modulo:  Vc ← Vc + Vd   no input
    // i.e Vc1 = Vc +Vd1
    #[allow(clippy::too_many_arguments)]
    pub fn s_vector_c1(
        s_vector_c1: Expression<F>,
        vector_m_c1: Expression<F>,
        vector_n_c1: Expression<F>,
        vector_o_c1: Expression<F>,
        vector_p_c1: Expression<F>,
        vector_carry_c1: Expression<F>,
        vector_m_c: Expression<F>,
        vector_n_c: Expression<F>,
        vector_o_c: Expression<F>,
        vector_p_c: Expression<F>,
        vector_m_d1: Expression<F>,
        vector_n_d1: Expression<F>,
        vector_o_d1: Expression<F>,
        vector_p_d1: Expression<F>,        
    ) -> Option<(&'static str, Expression<F>)> {
        let vector_c = vector_m_c + vector_n_c * F::from(1 << 32) + vector_o_c * F::from(1 << 64) + vector_p_c * F::from(1 << 96);
        let vector_d1 = vector_m_d1 + vector_n_d1 * F::from(1 << 32) + vector_o_d1 * F::from(1 << 64) + vector_p_d1 * F::from(1 << 96);
        let vector_sum = vector_c + vector_d1;

        let vector_c1 = vector_m_c1 + vector_n_c1 * F::from(1 << 32) + vector_o_c1 * F::from(1 << 64) + vector_p_c1 * F::from(1 << 96);

        let check = vector_sum - (vector_c1_carry * F::from(1 << 32)) -  vector_c1;

        Some(("vector_c1", s_vector_c1 * check))
    }

    // check if put spread_m_b2_lo_even
    // Fourth gate: Vb1 ← (Vb xor Vc1) >>> 24
    // 16,16,8,8,16
    // vector_b1 on abcd words
    #[allow(clippy::too_many_arguments)]
        pub fn s_vector_b1(
            s_vector_b1: Expression<F>,
            spread_m_b1_even: Expression<F>,
            spread_m_b1_odd: Expression<F>,
            spread_n_b1_even: Expression<F>,
            spread_n_b1_odd: Expression<F>,
            spread_o_b1_even: Expression<F>,
            spread_o_b1_odd: Expression<F>,
            spread_p_b1_even: Expression<F>,
            spread_p_b1_odd: Expression<F>,
            spread_m_c1: Expression<F>,
            spread_n_lo_c1: Expression<F>,
            spread_n_hi_c1: Expression<F>,
            spread_o_c1: Expression<F>,
            spread_p_c1: Expression<F>,
            spread_m_b: Expression<F>,
            spread_n_lo_b: Expression<F>,
            spread_n_hi_b: Expression<F>,
            spread_o_b: Expression<F>,
            spread_p_b: Expression<F>,
        ) -> Option<(&'static str, Expression<F>)> {

            let spread_witness = 
                spread_m_b1_even + spread_m_b1_odd * F::from(2)
                + (spread_n_b1_even + spread_n_b1_odd * F::from(2)) * F::from(1 << 32)
                + (spread_o_b1_even + spread_o_b1_odd * F::from(2)) * F::from(1 << 64)
                + (spread_p_b1_even + spread_p_b1_odd * F::from(2)) * F::from(1 << 96);

            // bit rotation
            let rot = spread_n_hi_c1.clone() + spread_n_hi_b.clone()
                + (spread_o_c1.clone() + spread_o_b.clone())* F::from(1 << 16)
                + (spread_p_c1.clone() + spread_p_b.clone())* F::from(1 << 48)
                + (spread_m_c1.clone() + spread_m_b.clone())* F::from(1 << 80)
                + (spread_n_lo_c1.clone() + spread_n_lo_b.clone())* F::from(1 << 112);

            let check = spread_witness + (rot * -F::ONE);

            Some(("vector_b1", s_vector_b1 * check))
        }

 
    // Fifth gate addition modulo, Va2 ← Va1 + Vb1 + y   with input
    #[allow(clippy::too_many_arguments)]
        pub fn s_vector_a2(
            s_vector_a2: Expression<F>,
            vector_m_a2: Expression<F>,
            vector_n_a2: Expression<F>,
            vector_o_a2: Expression<F>,
            vector_p_a2: Expression<F>,
            vector_carry_a2: Expression<F>,
            vector_m_a1: Expression<F>,
            vector_n_a1: Expression<F>,
            vector_o_a1: Expression<F>,
            vector_p_a1: Expression<F>,
            vector_m_b1: Expression<F>,
            vector_n_b1: Expression<F>,
            vector_o_b1: Expression<F>,
            vector_p_b1: Expression<F>,        
            vector_m_y: Expression<F>,
            vector_n_y: Expression<F>,
            vector_o_y: Expression<F>,
            vector_p_y: Expression<F>,
        ) -> Option<(&'static str, Expression<F>)> {
            let vector_a1 = vector_m_a1 + vector_n_a1 * F::from(1 << 32) + vector_o_a1 * F::from(1 << 64) + vector_p_a1 * F::from(1 << 96);
            let vector_b1 = vector_m_b1 + vector_n_b1 * F::from(1 << 32) + vector_o_b1 * F::from(1 << 64) + vector_p_b1 * F::from(1 << 96);
            let vector_y = vector_m_y + vector_n_y * F::from(1 << 32) + vector_o_y * F::from(1 << 64) + vector_p_y * F::from(1 << 96);
            let vector_sum = vector_a1 + vector_b1 + vector_y;

            let vector_a2 = vector_m_a2 + vector_n_a2 * F::from(1 << 16) + vector_o_a2 * F::from(1 << 32) + vector_p_a2 * F::from(1 << 48);

            let check = vector_sum (vector_c1_carry * F::from(1 << 32)) - vector_a2;

            Some(("vector_a2", s_vector_a2 * check))
        }


    // Sixth gate, xor and bit rotations: Vd2 ← (Vd1 xor Va2) >>> 16
    // vector_d1 on abcd words
    #[allow(clippy::too_many_arguments)]
    pub fn s_vector_d2(
        s_vector_d2: Expression<F>,
        spread_m_d2_even: Expression<F>,
        spread_m_d2_odd: Expression<F>,
        spread_n_d2_even: Expression<F>,
        spread_n_d2_odd: Expression<F>,
        spread_o_d2_even: Expression<F>,
        spread_o_d2_odd: Expression<F>,
        spread_p_d2_even: Expression<F>,
        spread_p_d2_odd: Expression<F>,
        spread_m_a2: Expression<F>,
        spread_n_a2: Expression<F>,
        spread_o_a2: Expression<F>,
        spread_p_a2: Expression<F>,
        spread_m_d1: Expression<F>,
        spread_n_d1: Expression<F>,
        spread_o_d1: Expression<F>,
        spread_p_d1: Expression<F>,
    ) -> Option<(&'static str, Expression<F>)> {

        let spread_witness = 
            spread_m_d2_even + spread_m_d2_odd * F::from(2)
            + (spread_n_d2_even + spread_n_d2_odd * F::from(2)) * F::from(1 << 32)
            + (spread_o_d2_even + spread_o_d2_odd * F::from(2)) * F::from(1 << 64)
            + (spread_p_d2_even + spread_p_d2_odd * F::from(2)) * F::from(1 << 96);


        // bit rotation
        let rot = spread_n_a2.clone() + spread_n_d1.clone()
            + (spread_o_a2.clone() + spread_o_d1.clone())* F::from(1 << 32)
            + (spread_p_a2.clone() + spread_p_d1.clone())* F::from(1 << 64)
            + (spread_m_a2.clone() + spread_m_d1.clone())* F::from(1 << 96);

        
        let check = spread_witness + (rot * -F::ONE);


        Some(("vector_d2", s_vector_d2 * check))

    }
        

    // Seventh gate addition modulo:  Vc2 ← Vc1 + Vd2   no input
    #[allow(clippy::too_many_arguments)]
    pub fn s_spread_c2(
        s_spread_c2: Expression<F>,
        spread_m_c2: Expression<F>,
        spread_n_c2: Expression<F>,
        spread_o_c2: Expression<F>,
        spread_p_c2: Expression<F>,
        spread_m_c1: Expression<F>,
        spread_n_c1: Expression<F>,
        spread_o_c1: Expression<F>,
        spread_p_c1: Expression<F>,
        spread_m_d2: Expression<F>,
        spread_n_d2: Expression<F>,
        spread_o_d2: Expression<F>,
        spread_p_d2: Expression<F>,        
    ) -> Option<(&'static str, Expression<F>)> {
        let spread_c1 = spread_m_c1 + spread_n_c1 * F::from(1 << 32) + spread_o_c1 * F::from(1 << 64) + spread_p_c1 * F::from(1 << 96);
        let spread_d2 = spread_m_d2 + spread_n_d2 * F::from(1 << 32) + spread_o_d2 * F::from(1 << 64) + spread_p_d2 * F::from(1 << 96);
        let spread_sum = spread_c1 + spread_d2;

        let spread_c2 = spread_m_c2 + spread_n_c2 * F::from(1 << 32) + spread_o_c2 * F::from(1 << 64) + spread_p_c2 * F::from(1 << 96);

        let check = spread_sum - spread_c2;

        Some(("spread_c2", s_spread_c2 * (spread_sum - spread_c2)))
    }

    // 1+15+16+16+16
    // Eight gate: Vb2 ← (Vb1 xor Vc2) >>> 63
    // vector_b1 on abcd words
    #[allow(clippy::too_many_arguments)]
        pub fn s_vector_b2(
            s_vector_b2: Expression<F>,
            spread_m_b2_even: Expression<F>,
            spread_m_b2_odd: Expression<F>,
            spread_n_b2_even: Expression<F>,
            spread_n_b2_odd: Expression<F>,
            spread_o_b2_even: Expression<F>,
            spread_o_b2_odd: Expression<F>,
            spread_p_b2_even: Expression<F>,
            spread_p_b2_odd: Expression<F>,
            spread_m_c2: Expression<F>,
            spread_n_c2: Expression<F>,
            spread_o_c2: Expression<F>,
            spread_p_lo_c2: Expression<F>,
            spread_p_hi_c2: Expression<F>,
            spread_m_b1: Expression<F>,
            spread_n_b1: Expression<F>,
            spread_o_b1: Expression<F>,
            spread_p_lo_b1: Expression<F>,
            spread_p_hi_b1: Expression<F>,
        ) -> Option<(&'static str, Expression<F>)> {

            let spread_witness = 
                spread_m_b2_even + spread_m_b2_odd * F::from(2)
                + (spread_n_b2_even + spread_n_b2_odd * F::from(2)) * F::from(1 << 32)
                + (spread_o_b2_even + spread_o_b2_odd * F::from(2)) * F::from(1 << 64)
                + (spread_p_b2_even + spread_p_b2_odd * F::from(2)) * F::from(1 << 96);


            // bit rotation
            let rot = spread_p_hi_c2.clone() + spread_p_hi_b1.clone()
                + (spread_m_c2.clone() + spread_m_b1.clone())* F::from(1 << 2)
                + (spread_n_c2.clone() + spread_n_b1.clone())* F::from(1 << 34)
                + (spread_o_c2.clone() + spread_o_b1.clone())* F::from(1 << 66)
                + (spread_p_lo_c2.clone() + spread_p_lo_b1.clone())* F::from(1 << 98);


            let check = spread_witness + (rot * -F::ONE);


            Some(("vector_b2", s_vector_b2 * check))

        }

    // todo 
    // s_digest on final round
    #[allow(clippy::too_many_arguments)]
    pub fn s_digest(
        s_digest: Expression<F>,
        lo_0: Expression<F>,
        hi_0: Expression<F>,
        word_0: Expression<F>,
        lo_1: Expression<F>,
        hi_1: Expression<F>,
        word_1: Expression<F>,
        lo_2: Expression<F>,
        hi_2: Expression<F>,
        word_2: Expression<F>,
        lo_3: Expression<F>,
        hi_3: Expression<F>,
        word_3: Expression<F>,
    ) -> impl IntoIterator<Item = Constraint<F>> {
        let check_lo_hi = |lo: Expression<F>, hi: Expression<F>, word: Expression<F>| {
            lo + hi * F::from(1 << 16) - word
        };

        Constraints::with_selector(
            s_digest,
            [
                ("check_lo_hi_0", check_lo_hi(lo_0, hi_0, word_0)),
                ("check_lo_hi_1", check_lo_hi(lo_1, hi_1, word_1)),
                ("check_lo_hi_2", check_lo_hi(lo_2, hi_2, word_2)),
                ("check_lo_hi_3", check_lo_hi(lo_3, hi_3, word_3)),
            ],
        )
    }

    }


