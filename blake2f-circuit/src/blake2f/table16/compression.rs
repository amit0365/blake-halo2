//todo fix xor gates rotation gates and word lo,mo,el,hi

mod compression_gate;
mod subregion_initial;
mod compression_util;


use halo2_proofs::{
    circuit::{Layouter, Value},
    pasta::pallas,
    plonk::{Advice, Column, Constraint, Constraints, ConstraintSystem, Error, Expression, Selector, Instance},
    poly::Rotation,
};

use group::ff::{Field, PrimeField};
//use pasta_curves::arithmetic::Field;
use std::{marker::PhantomData, ops::Range};

use crate::blake2f::table16::{AssignedBits, BlockWord, SpreadInputs, SpreadVar, Table16Assignment};
use crate::blake2f::DIGEST_SIZE;


//use pasta_curves::pallas;

const ROUNDS: usize = 12;
const STATE: usize = 8;

use compression_gate::MixingGate;

use crate::blake2f::table16::{util::{i2lebsp, lebs2ip}};

//use super::{spread_table::{SpreadVar, SpreadInputs}};
// utils::{i2lebsp, lebs2ip}, , table16::{self, Table16Assignment}
//use super::spread_table::AssignedBits;

pub trait VectorVar<
    const A_LEN: usize,
    const B_LEN: usize,
    const C_LEN: usize,
    const D_LEN: usize,
>
{
    fn spread_m(&self) -> Value<[bool; A_LEN]>;
    fn spread_n(&self) -> Value<[bool; B_LEN]>;
    fn spread_o(&self) -> Value<[bool; C_LEN]>;
    fn spread_p(&self) -> Value<[bool; D_LEN]>;

    fn xor_v(&self) -> Value<[bool; 64]> {
        self.spread_m()
            .zip(self.spread_n())
            .zip(self.spread_o())
            .zip(self.spread_p())
            .map(|(((m, n), o), p)| {
                
                let xor = m
                    .iter()
                    .chain(n.iter())
                    .chain(o.iter())
                    .chain(p.iter())
                    .copied()
                    .collect::<Vec<_>>();

                let xor = lebs2ip::<64>(&xor.try_into().unwrap());

                i2lebsp(xor)
            })
    }
}

/// A variable that represents the `[A,B,C,D]` words of the SHA-256 internal state.
///
/// The structure of this variable is influenced by the following factors:
/// - In `Σ_0(A)` we need `A` to be split into pieces `(a,b,c,d)` of lengths `(2,11,9,10)`
///   bits respectively (counting from the little end), as well as their spread forms.
/// - `Maj(A,B,C)` requires having the bits of each input in spread form. For `A` we can
///   reuse the pieces from `Σ_0(A)`. Since `B` and `C` are assigned from `A` and `B`
///   respectively in each round, we therefore also have the same pieces in earlier rows.
///   We align the columns to make it efficient to copy-constrain these forms where they
///   are needed.
#[derive(Clone, Debug)]
pub struct AbcdVar {
    m: SpreadVar<16, 32>,
    n: SpreadVar<16, 32>,
    o: SpreadVar<16, 32>,
    p: SpreadVar<16, 32>,
}

impl AbcdVar {
    fn m_range() -> Range<usize> {
        0..16
    }

    fn n_range() -> Range<usize> {
        16..32
    }

    fn o_range() -> Range<usize> {
        32..48
    }

    fn p_range() -> Range<usize> {
        48..64
    }

    fn pieces(val: u64) -> Vec<Vec<bool>> {
        let val: [bool; 64] = i2lebsp(val.into());
        vec![
            val[Self::m_range()].to_vec(),
            val[Self::n_range()].to_vec(),
            val[Self::o_range()].to_vec(),
            val[Self::p_range()].to_vec(),
        ]
    }
}

impl VectorVar<32, 32, 32, 32> for AbcdVar {
    fn spread_m(&self) -> Value<[bool; 32]> {
        self.m.spread.value().map(|v| v.0)
    }

    fn spread_n(&self) -> Value<[bool; 32]> {
        self.n.spread.value().map(|v| v.0)
    }

    fn spread_o(&self) -> Value<[bool; 32]> {
        self.o.spread.value().map(|v| v.0)
    }

    fn spread_p(&self) -> Value<[bool; 32]> {
        self.p.spread.value().map(|v| v.0)
    }
}

#[derive(Clone, Debug)]
pub struct EfghVar {
    m: SpreadVar<16, 32>,
    n_lo: SpreadVar<8, 16>,
    n_hi: SpreadVar<8, 16>,
    o: SpreadVar<16, 32>,
    p: SpreadVar<16, 32>,
}

impl EfghVar {
    fn m_range() -> Range<usize> {
        0..16
    }

    fn n_lo_range() -> Range<usize> {
        16..24
    }

    fn n_hi_range() -> Range<usize> {
        24..32
    }

    fn o_range() -> Range<usize> {
        32..48
    }

    fn p_range() -> Range<usize> {
        48..64
    }

    fn pieces(val: u64) -> Vec<Vec<bool>> {
        let val: [bool; 64] = i2lebsp(val.into());
        vec![
            val[Self::m_range()].to_vec(),
            val[Self::n_lo_range()].to_vec(),
            val[Self::n_hi_range()].to_vec(),
            val[Self::o_range()].to_vec(),
            val[Self::p_range()].to_vec(),
        ]
    }
}


impl VectorVar<32,32,32,32> for EfghVar {

    fn spread_m(&self) -> Value<[bool; 32]> {
        self.m.spread.value().map(|v| v.0)
    }

    fn spread_n(&self) -> Value<[bool; 32]> {
        self.n_lo
            .spread
            .value()
            .zip(self.n_hi.spread.value())
            .map(|(n_lo, b_hi)| {
                n_lo.iter()
                    .chain(b_hi.iter())
                    .copied()
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
    }

    fn spread_o(&self) -> Value<[bool; 32]> {
        self.o.spread.value().map(|v| v.0)
    }

    fn spread_p(&self) -> Value<[bool; 32]> {
        self.p.spread.value().map(|v| v.0)
    }

}

#[derive(Clone, Debug)]
pub struct IjklVar {
    m: SpreadVar<16, 32>,
    n: SpreadVar<16, 32>,
    o: SpreadVar<16, 32>,
    p_lo: SpreadVar<1, 2>,
    p_hi: SpreadVar<15, 32>,
}

impl IjklVar {
    
    fn m_range() -> Range<usize> {
        1..16
    }

    fn n_range() -> Range<usize> {
        16..32
    }

    fn o_range() -> Range<usize> {
        32..48
    }


    fn p_lo_range() -> Range<usize> {
        48..49
    }

    fn p_hi_range() -> Range<usize> {
        49..64
    }

    fn pieces(val: u64) -> Vec<Vec<bool>> {
        let val: [bool; 64] = i2lebsp(val.into());
        vec![
            val[Self::n_range()].to_vec(),
            val[Self::o_range()].to_vec(),
            val[Self::p_lo_range()].to_vec(),
            val[Self::p_hi_range()].to_vec(),
            val[Self::m_range()].to_vec(),
        ]
    }
}


impl VectorVar<32,32,32,32> for IjklVar {

    fn spread_m(&self) -> Value<[bool; 32]> {
        self.m.spread.value().map(|v| v.0)
    }

    fn spread_n(&self) -> Value<[bool; 32]> {
        self.n.spread.value().map(|v| v.0)
    }

    fn spread_o(&self) -> Value<[bool; 32]> {
        self.o.spread.value().map(|v| v.0)
    }

    fn spread_p(&self) -> Value<[bool; 32]> {
        self.p_lo
            .spread
            .value()
            .zip(self.p_hi.spread.value())
            .map(|(p_lo, p_hi)| {
                p_lo.iter()
                    .chain(p_hi.iter())
                    .copied()
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
    }

}


#[derive(Clone, Debug)]
pub struct RoundWordDense(AssignedBits<16>, AssignedBits<16>,AssignedBits<16>, AssignedBits<16>);

impl From<(AssignedBits<16>, AssignedBits<16>,AssignedBits<16>, AssignedBits<16>)> for RoundWordDense {
    fn from(halves: (AssignedBits<16>, AssignedBits<16>,AssignedBits<16>, AssignedBits<16>)) -> Self {
        Self(halves.0, halves.1, halves.2, halves.3)
    }
}

impl RoundWordDense {
    pub fn value(&self) -> Value<u64> {
        self.0
            .value_u16()
            .zip(self.1.value_u16())
            .zip(self.2.value_u16())
            .zip(self.3.value_u16())
            .map(|(((lo, mo),el),hi)| lo as u64 + (1 << 16) * mo as u64 + (1 << 32) * el as u64 + (1 << 48) * hi as u64)
    }
}

#[derive(Clone, Debug)]
pub struct RoundWordSpread(AssignedBits<32>, AssignedBits<32>,AssignedBits<32>, AssignedBits<32>);


impl From<(AssignedBits<32>, AssignedBits<32>, AssignedBits<32>, AssignedBits<32>)> for RoundWordSpread {
    fn from(halves: (AssignedBits<32>, AssignedBits<32>, AssignedBits<32>, AssignedBits<32>)) -> Self {
        Self(halves.0, halves.1, halves.2, halves.3)
    }
}

impl RoundWordSpread {
    pub fn value(&self) -> Value<u128> {
        self.0
            .value_u32()
            .zip(self.1.value_u32())
            .zip(self.1.value_u32())
            .zip(self.1.value_u32())
            .map(|(((lo, mo),el),hi)| lo as u128 + mo as u128 + el as u128 + (1 << 32) * hi as u128)
    }
}

#[derive(Clone, Debug)]
pub struct RoundWordA {
    pieces: Option<AbcdVar>,
    dense_halves: RoundWordDense,
    spread_halves: Option<RoundWordSpread>,
}

impl RoundWordA {
    pub fn new(
        pieces: AbcdVar,
        dense_halves: RoundWordDense,
        spread_halves: RoundWordSpread,
    ) -> Self {
        RoundWordA {
            pieces: Some(pieces),
            dense_halves,
            spread_halves: Some(spread_halves),
        }
    }

    pub fn new_dense(dense_halves: RoundWordDense) -> Self {
        RoundWordA {
            pieces: None,
            dense_halves,
            spread_halves: None,
        }
    }
}


#[derive(Clone, Debug)]
pub struct RoundWord {
    dense_halves: RoundWordDense,
    spread_halves: RoundWordSpread,
}

impl RoundWord {
    pub fn new(dense_halves: RoundWordDense, spread_halves: RoundWordSpread) -> Self {
        RoundWord {
            dense_halves,
            spread_halves,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstantWord {
    dense_halves: ConstantWordDense,
    spread_halves: ConstantWordSpread,
}

impl ConstantWord {
    pub fn new(dense_halves: ConstantWordDense, spread_halves: ConstantWordSpread) -> Self {
        ConstantWord {
            dense_halves,
            spread_halves,
        }
    }
}

/// The internal persistent state for Blake.
#[derive(Clone, Debug)]
pub struct State {
    v0: Option<StateWord>,
    v1: Option<StateWord>,
    v2: Option<StateWord>,
    v3: Option<StateWord>,
    v4: Option<StateWord>,
    v5: Option<StateWord>,
    v6: Option<StateWord>,
    v7: Option<StateWord>,
    v8: Option<StateWord>,
    v9: Option<StateWord>,
    v10: Option<StateWord>,
    v11: Option<StateWord>,
    v12: Option<StateWord>,
    v13: Option<StateWord>,
    v14: Option<StateWord>,
    v15: Option<StateWord>,
}

impl State {
    #[allow(clippy::many_single_char_names)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        v0: StateWord,
        v1: StateWord,
        v2: StateWord,
        v3: StateWord,
        v4: StateWord,
        v5: StateWord,
        v6: StateWord,
        v7: StateWord,
        v8: StateWord,
        v9: StateWord,
        v10: StateWord,
        v11: StateWord,
        v12: StateWord,
        v13: StateWord,
        v14: StateWord,
        v15: StateWord,
    ) -> Self {
        State {
            v0: Some(v0),
            v1: Some(v1),
            v2: Some(v2),
            v3: Some(v3),
            v4: Some(v4),
            v5: Some(v5),
            v6: Some(v6),
            v7: Some(v7),
            v8: Some(v8),
            v9: Some(v9),
            v10: Some(v10),
            v11: Some(v11),
            v12: Some(v12),
            v13: Some(v13),
            v14: Some(v14),
            v15: Some(v15),
        }
    }

    pub fn empty_state() -> Self {
        State {
            v0: None,
            v1: None,
            v2: None,
            v3: None,
            v4: None,
            v5: None,
            v6: None,
            v7: None,
            v8: None,
            v9: None,
            v10: None,
            v11: None,
            v12: None,
            v13: None,
            v14: None,
            v15: None,
        }
    }
}


// todo check why different objects for ABC and see how impllemented
#[derive(Clone, Debug)]
pub enum StateWord {
    V0(RoundWord),
    V1(RoundWord),
    V2(RoundWord),
    V3(RoundWord),
    V4(RoundWord),
    V5(RoundWord),
    V6(RoundWord),
    V7(RoundWord),
    V8(ConstantWord),
    V9(ConstantWord),
    V10(ConstantWord),
    V11(ConstantWord),
    V12(ConstantWord),
    V13(ConstantWord),
    V14(ConstantWord),
    V15(ConstantWord),
}

#[derive(Clone, Debug)]
pub(crate) struct MixingConfig {
    lookup: SpreadInputs,
    extras: [Column<Advice>; 6],
    s_vector_a1: Selector,
    s_vector_b1: Selector,
    s_vector_c1: Selector,
    s_vector_d1: Selector,
    s_vector_a2: Selector,
    s_vector_b2: Selector,
    s_vector_c2: Selector,
    s_vector_d2: Selector,

    // Decomposition gate for AbcdVar
    s_decompose_abcd: Selector,
    // Decomposition gate for EfghVar
    s_decompose_efgh: Selector,
    // Decomposition gate for IjklVar
    s_decompose_ijkl: Selector,

    s_digest: Selector,

}

impl Table16Assignment for MixingConfig {}

// implements mixing function
impl MixingConfig {
    pub(super) fn configure(
        meta: &mut ConstraintSystem<pallas::Base>,
        lookup: SpreadInputs,
        extras: [Column<Advice>; 6],
    ) -> Self {
        let s_vector_a1 = meta.selector();
        let s_vector_b1 = meta.selector();
        let s_vector_c1 = meta.selector();
        let s_vector_d1 = meta.selector();
        let s_vector_a2 = meta.selector();
        let s_vector_b2 = meta.selector();
        let s_vector_c2 = meta.selector();
        let s_vector_d2 = meta.selector();

        // (16, 16, 16, 16)-bit chunks
        let s_decompose_abcd = meta.selector();

        // (16, 16, 8, 8, 16)-bit chunks
        let s_decompose_efgh = meta.selector();

        // (1, 15, 16, 16, 16)-bit chunks
        let s_decompose_ijkl = meta.selector();

        let s_digest = meta.selector();

        // Rename these here for ease of matching the gates to the specification.
        let a_0 = lookup.tag;
        let a_1 = lookup.dense;
        let a_2 = lookup.spread;
        let a_3 = lookup.dense;
        let a_4 = lookup.spread;
        let a_5 = extras[0];
        let a_6 = extras[1];
        let a_7 = extras[2];
        let a_8 = extras[3];
        let a_9 = extras[4];


        // todo if don't constrain tag how lookup work ? is it only {a1,a2}
        // do we need addition modulo in blake2b -- YES ITS MOD 2^64
        // todo make the circuit wider to reduce FFTs, this would increase proof size due to more poly commit

        // Decompose `A,B,C,D` words into (16, 16, 16, 16)-bit chunks.

        // Structure should be tag m, m , s_m as we lookup tuple a0, a1, a2
        // no need to constrain tag for 16 bits

        //    a0   |  a1  |  a2    |  a3   |   a4   |    a5   |   a6    |    a7    |    a8    |
        // ------- |  a   |  s_a   |   c   |  s_c   |   w_lo  |  s_w_lo |    w_el  |  s_w_el  |
        // ------- |  b   |  s_b   |   d   |  s_d   |   w_mo  |  s_w_mo |    w_hi  |  s_w_hi  |

        // Decompose `A,B,C,D` words into (16, 16, 16, 16)-bit chunks.
        // VirtualCells API why ignoring _?
        meta.create_gate("decompose ABCD", |meta| {

            let s_decompose_abcd = meta.query_selector(s_decompose_abcd);

            let a = meta.query_advice(a_1, Rotation::prev()); // 16-bit chunk
            let spread_a = meta.query_advice(a_2, Rotation::prev());

            let b = meta.query_advice(a_1, Rotation::cur()); //  16-bit chunk
            let spread_b = meta.query_advice(a_2, Rotation::cur());

            let c = meta.query_advice(a_3, Rotation::prev()); // 16-bit chunk
            let spread_c = meta.query_advice(a_4, Rotation::prev());

            let d = meta.query_advice(a_4, Rotation::cur()); // 16-bit chunk
            let spread_d = meta.query_advice(a_5, Rotation::cur());


            // todo how are these calculated -- look fn assign_word_halves_dense
            let word_lo = meta.query_advice(a_5, Rotation::prev());
            let spread_word_lo = meta.query_advice(a_6, Rotation::prev());
            let word_mo = meta.query_advice(a_5, Rotation::cur());
            let spread_word_mo = meta.query_advice(a_6, Rotation::cur());
            let word_el = meta.query_advice(a_7, Rotation::prev());
            let spread_word_el = meta.query_advice(a_8, Rotation::prev());
            let word_hi = meta.query_advice(a_7, Rotation::cur());
            let spread_word_hi = meta.query_advice(a_8, Rotation::cur());

            MixingGate::s_decompose_abcd(
                s_decompose_abcd,
                a,
                spread_a,
                b,
                spread_b,
                c,
                spread_c,
                d,
                spread_d,
                word_lo,
                spread_word_lo,
                word_mo,
                spread_word_mo,
                word_el,
                spread_word_el,
                word_hi,
                spread_word_hi,
            )
        });


        //     a0      |  a1     |  a2       |  a3    |  a4     |  a5    |  a6      |  a7    |  a8      |
        //   -------   |  e      |  s_e      |        |         |  w_lo  |  s_w_lo  |  w_el  |  s_w_el  |
        //     {0}     |  f_lo   |  s_f_lo   |  f_hi  |  s_f_hi |  w_mo  |  s_w_mo  |  w_hi  |  s_w_hi  |
        //   -------   |  g      |  s_g      |   h    |  s_h    |  ---   |   ---    |  ---   |   ---    |


        // Decompose `H,G,F,E` words into (16, 16, 8, 8, 16)-bit chunks.
        meta.create_gate("Decompose EFGH", |meta| {

            let s_decompose_efgh = meta.query_selector(s_decompose_efgh);

            let e = meta.query_advice(a_1, Rotation::prev()); // 16-bit chunk
            let spread_e = meta.query_advice(a_2, Rotation::prev());

            let tag_f_lo = meta.query_advice(a_0, Rotation::cur()); // 8-bit chunk
            let f_lo = meta.query_advice(a_1, Rotation::cur()); // 8-bit chunk
            let spread_f_lo = meta.query_advice(a_2, Rotation::cur());

            let tag_f_hi = meta.query_advice(a_0, Rotation::cur()); // 8-bit chunk
            let f_hi = meta.query_advice(a_3, Rotation::next()); // 8-bit chunk
            let spread_f_hi = meta.query_advice(a_4, Rotation::next());

            let g: Expression<pasta_curves::Fp> = meta.query_advice(a_1, Rotation::next()); // 16-bit chunk
            let spread_g = meta.query_advice(a_2, Rotation::next());

            let h = meta.query_advice(a_3, Rotation::next()); // 16-bit chunk
            let spread_h = meta.query_advice(a_4, Rotation::next());

            let word_lo = meta.query_advice(a_5, Rotation::prev());
            let spread_word_lo = meta.query_advice(a_6, Rotation::prev());
            let word_mo = meta.query_advice(a_5, Rotation::cur());
            let spread_word_mo = meta.query_advice(a_6, Rotation::cur());
            let word_el = meta.query_advice(a_7, Rotation::prev());
            let spread_word_el = meta.query_advice(a_8, Rotation::prev());
            let word_hi = meta.query_advice(a_7, Rotation::cur());
            let spread_word_hi = meta.query_advice(a_8, Rotation::cur());

            MixingGate::s_decompose_efgh(
                s_decompose_efgh,
                e,
                spread_e,
                tag_f_lo,
                f_lo,
                spread_f_lo,
                tag_f_hi,
                f_hi,
                spread_f_hi,
                g,
                spread_g,
                h,
                spread_h,
                word_lo,
                spread_word_lo,
                word_mo,
                spread_word_mo,
                word_el,
                spread_word_el,
                word_hi,
                spread_word_hi,
            )
        });

        // Decompose `I,J,K,L` words into (1, 15, 16, 16)-bit chunks.
        // add range check for 1 bit chunk, does not require lookup
        //    a0      |  a1     |  a2       |  a3    |  a4     |  a5    |  a6      |  a7      |  a8       |
        //   {0,1}    |  i_hi   |  s_i_hi   |        |         |  w_lo  |  s_w_lo  |  w_el    |  s_w_el   |
        //            |   j     |  s_j      |   k    |  s_k    |  w_mo  |  s_w_mo  |  w_hi    |  s_w_hi   |
        //            |   l     |  s_l      |        |         |  i_lo  |  s_i_lo  |          |           |


        // Decompose `L,K,J,I` words into (16, 16, 16, 15, 1)-bit chunks.
        meta.create_gate("Decompose IJKL", |meta| {

            let s_decompose_ijkl = meta.query_selector(s_decompose_ijkl);

            let i_lo = meta.query_advice(a_5, Rotation::prev()); // 1-bit chunk
            let spread_i_lo = meta.query_advice(a_6, Rotation::prev());

            let tag_i_hi = meta.query_advice(a_0, Rotation::prev()); // 15-bit chunk
            let i_hi = meta.query_advice(a_1, Rotation::prev()); // 15-bit chunk
            let spread_i_hi = meta.query_advice(a_2, Rotation::prev());

            let j = meta.query_advice(a_1, Rotation::cur()); // 16-bit chunk
            let spread_j = meta.query_advice(a_2, Rotation::cur());

            let k = meta.query_advice(a_3, Rotation::cur()); // 16-bit chunk
            let spread_k = meta.query_advice(a_4, Rotation::cur());

            let l = meta.query_advice(a_1, Rotation::next()); // 16-bit chunk
            let spread_l = meta.query_advice(a_2, Rotation::next());


            let word_lo = meta.query_advice(a_5, Rotation::prev());
            let spread_word_lo = meta.query_advice(a_6, Rotation::prev());
            let word_mo = meta.query_advice(a_5, Rotation::cur());
            let spread_word_mo = meta.query_advice(a_6, Rotation::cur());
            let word_el = meta.query_advice(a_7, Rotation::prev());
            let spread_word_el = meta.query_advice(a_8, Rotation::prev());
            let word_hi = meta.query_advice(a_7, Rotation::cur());
            let spread_word_hi = meta.query_advice(a_8, Rotation::cur());

            MixingGate::s_decompose_ijkl(
                s_decompose_ijkl,
                i_lo,
                spread_i_lo,
                tag_i_hi,
                i_hi,
                spread_i_hi,
                j,
                spread_j,
                k,
                spread_k,
                l,
                spread_l,
                word_lo,
                spread_word_lo,
                word_mo,
                spread_word_mo,
                word_el,
                spread_word_el,
                word_hi,
                spread_word_hi,
            )
        });

        // The operand and result chunks can be constrained using spread, by looking up each chunk in the "dense" column within a subset of the table.
        // This way we additionally get the "spread" form of the output for free, use in the addition gates?? figure from E_new gate ? decompose_a
        // colns a1,a2 are defined to carry spread outputs for the result chunks in table16, might need to change gate layout

        // carry must be constrained to the precise range of allowed carry values for the number of operands. 
        // We do this with a small range constraint.?? figure from E_new gate
        // Va1 = Va + Vb + x abcd words

        //     a0     |   a1    |    a2     |  a3    |  a4    |  a5    |  a6   |  a7    |  a8     |  a9    |
        //   -------  |  m_a1   |  s_m_a1   |  n_a1  | s_n_a1 |  m_a   |  n_a  |  o_a   |  p_a    |  c_a1  | 
        //   -------  |  o_a1   |  s_o_a1   |  p_a1  | s_p_a1 |  m_b   |  n_b  |  o_b   |  p_b    |        |
        //            |         |           |        |        |  m_x   |  n_x  |  o_x   |  p_x    |        |


        // Va1 = Va + Vb + x abcd words
        // (16, 16, 16, 16)-bit into ( m, n, o, p ) chunks

        meta.create_gate("s_vector_a1", |meta| {

            let s_vector_a1 = meta.query_selector(s_vector_a1);

            let vector_m_a1 = meta.query_advice(a_1, Rotation::prev());
            let vector_s_m_a1 = meta.query_advice(a_2, Rotation::prev());

            let vector_n_a1 = meta.query_advice(a_3, Rotation::prev());
            let vector_s_n_a1 = meta.query_advice(a_4, Rotation::prev());

            let vector_o_a1 = meta.query_advice(a_1, Rotation::cur());
            let vector_s_o_a1 = meta.query_advice(a_2, Rotation::cur());

            let vector_p_a1 = meta.query_advice(a_3, Rotation::cur());
            let vector_s_p_a1 = meta.query_advice(a_4, Rotation::cur());

            let vector_carry_a1 = meta.query_advice(a_9, Rotation::prev());

            let vector_m_a = meta.query_advice(a_5, Rotation::prev());
            let vector_n_a = meta.query_advice(a_6, Rotation::prev());
            let vector_o_a = meta.query_advice(a_7, Rotation::prev());
            let vector_p_a = meta.query_advice(a_8, Rotation::prev());

            let vector_m_b = meta.query_advice(a_5, Rotation::cur());
            let vector_n_b = meta.query_advice(a_6, Rotation::cur());
            let vector_o_b = meta.query_advice(a_7, Rotation::cur());
            let vector_p_b = meta.query_advice(a_8, Rotation::cur());

            let vector_m_x = meta.query_advice(a_5, Rotation::next());
            let vector_n_x = meta.query_advice(a_6, Rotation::next());
            let vector_o_x = meta.query_advice(a_7, Rotation::next());
            let vector_p_x = meta.query_advice(a_8, Rotation::next());


            MixingGate::s_vector_a1(
                s_vector_a1,
                vector_m_a1,
                vector_s_m_a1,
                vector_n_a1,
                vector_s_n_a1,
                vector_o_a1,
                vector_s_o_a1,
                vector_p_a1,
                vector_s_p_a1,
                vector_carry_a1,
                vector_m_a,
                vector_n_a,
                vector_o_a,
                vector_p_a,
                vector_m_b,
                vector_n_b,
                vector_o_b,
                vector_p_b,
                vector_m_x,
                vector_n_x,
                vector_o_x,
                vector_p_x,
            )
        });
 
    // todo why the upper sigma 0, 1 gates have only spread vals in rounds while the dense vals in gates section 
    // keep the output of xor in spread and then do rotation
    // Vd1 ← (Vd xor Va1) rotate right 32

    // do we range check even and odd bits? yes to prevent overflow and it also gives spread

    //     a0   |   a1        |    a2         |     a3    |      a4    |   a5      |     a6     |    a7    |   a8     |   a9    |
    //  {0,1,2} |  m_d1_even  |  s_m_d1_even  |  m_d1_odd | s_m_d1_odd | p_d1_even | s_p_d1_odd |  s_m_d   |  s_o_d   |   ----  |
    //  {0,1,2} |  n_d1_even  |  s_n_d1_even  |  n_d1_odd | s_n_d1_odd | p_d1_odd  | s_p_d1_odd |  s_n_d   |  s_p_d   |   ----  |
    //  {0,1,2} |  o_d1_even  |  s_o_d1_even  |  o_d1_odd | s_o_d1_odd | s_m_a1    |  s_n_a1    |  s_o_a1  |  s_p_a1  |   ----  |
    //  {0,1,2} |  o_d1_even  |  s_o_d1_even  |  o_d1_odd | s_o_d1_odd | s_m_a1    |  s_n_a1    |  s_o_a1  |  s_p_a1  |   ----  |

        // Vd1 ← (Vd xor Va1) rotate right 32
        // (16, 16, 16, 16)-bit chunks
        meta.create_gate("s_vector_d1", |meta| {

            let s_spread_d1 = meta.query_selector(s_spread_d1);

            let spread_m_d1_even = meta.query_advice(a_2, Rotation::prev());
            let spread_m_d1_odd = meta.query_advice(a_2, Rotation::cur());
            let spread_n_d1_even = meta.query_advice(a_2, Rotation::next());
            let spread_n_d1_odd = meta.query_advice(a_2, Rotation(2));

            let spread_o_d1_even = meta.query_advice(a_2, Rotation(3));
            let spread_o_d1_odd = meta.query_advice(a_2, Rotation(4));
            let spread_p_d1_even = meta.query_advice(a_2, Rotation(5));
            let spread_p_d1_odd = meta.query_advice(a_2, Rotation(6));

            let spread_m_a1 = meta.query_advice(a_4, Rotation::prev());
            let spread_n_a1 = meta.query_advice(a_4, Rotation::cur());
            let spread_o_a1 = meta.query_advice(a_4, Rotation::next());
            let spread_p_a1 = meta.query_advice(a_4, Rotation(2));

            let spread_m_d = meta.query_advice(a_5, Rotation::prev());
            let spread_n_d = meta.query_advice(a_5, Rotation::cur());
            let spread_o_d = meta.query_advice(a_5, Rotation::next());
            let spread_p_d = meta.query_advice(a_5, Rotation(2));


            MixingGate::s_spread_d1(
                s_spread_d1,
                spread_m_d1_even,
                spread_m_d1_odd,
                spread_n_d1_even,
                spread_n_d1_odd,
                spread_o_d1_even,
                spread_o_d1_odd,
                spread_p_d1_even,
                spread_p_d1_odd,
                spread_m_a1,
                spread_n_a1,
                spread_o_a1,
                spread_p_a1,
                spread_m_d,
                spread_n_d,
                spread_o_d,
                spread_p_d,
            )
        });

        // Vc1 = Vc + Vd1 abcd words

        //     a0     |   a1    |    a2     |  a3    |  a4    |  a5    |  a6   |
        //  tag_m_a1  |  m_a1   |  s_m_a1   |a1_carry|  m_a   |  m_b   |  ---  |  ----   |
        //  tag_n_a1  |  n_a1   |  s_m_a1   |        |  n_a   |  n_b   |  ---  |  ----   |
        //  tag_o_a1  |  o_a1   |  s_m_a1   |        |  o_a   |  o_b   |  ---  |  ----   |
        //  tag_p_a1  |  p_a1   |  s_m_a1   |        |  p_a   |  p_b   |  ---  |  ----   |

        // Vc1=Vc+Vd1 abcd words
        // (16, 16, 16, 16)-bit chunks
        meta.create_gate("s_vector_c1", |meta| {
            let s_vector_c1 = meta.query_selector(s_vector_c1);
            let vector_m_c1 = meta.query_advice(a_1, Rotation::prev());
            let vector_n_c1 = meta.query_advice(a_1, Rotation::cur());
            let vector_o_c1 = meta.query_advice(a_1, Rotation::next());
            let vector_p_c1 = meta.query_advice(a_1, Rotation(2));

            let vector_m_c = meta.query_advice(a_4, Rotation::prev());
            let vector_n_c = meta.query_advice(a_4, Rotation::cur());
            let vector_o_c = meta.query_advice(a_4, Rotation::next());
            let vector_p_c = meta.query_advice(a_4, Rotation(2));

            let vector_m_d1 = meta.query_advice(a_5, Rotation::prev());
            let vector_n_d1 = meta.query_advice(a_5, Rotation::cur());
            let vector_o_d1 = meta.query_advice(a_5, Rotation::next());
            let vector_p_d1 = meta.query_advice(a_5, Rotation(2));


            MixingGate::s_vector_c1(
                s_vector_c1,
                vector_m_c1,
                vector_n_c1,
                vector_o_c1,
                vector_p_c1,
                vector_m_c,
                vector_n_c,
                vector_o_c,
                vector_p_c,
                vector_m_d1,
                vector_n_d1,
                vector_o_d1,
                vector_p_d1,

            )
        });

        // s_ch_neg on efgh words
        // Second part of Choice gate on (E, F, G), ¬E ∧ G
        meta.create_gate("s_ch_neg", |meta| {
            let s_ch_neg = meta.query_selector(s_ch_neg);
            let spread_q0_even = meta.query_advice(a_2, Rotation::prev());
            let spread_q0_odd = meta.query_advice(a_2, Rotation::cur());
            let spread_q1_even = meta.query_advice(a_2, Rotation::next());
            let spread_q1_odd = meta.query_advice(a_3, Rotation::cur());
            let spread_e_lo = meta.query_advice(a_5, Rotation::prev());
            let spread_e_hi = meta.query_advice(a_5, Rotation::cur());
            let spread_e_neg_lo = meta.query_advice(a_3, Rotation::prev());
            let spread_e_neg_hi = meta.query_advice(a_4, Rotation::prev());
            let spread_g_lo = meta.query_advice(a_3, Rotation::next());
            let spread_g_hi = meta.query_advice(a_4, Rotation::next());

            MixingGate::s_ch_neg(
                s_ch_neg,
                spread_q0_even,
                spread_q0_odd,
                spread_q1_even,
                spread_q1_odd,
                spread_e_lo,
                spread_e_hi,
                spread_e_neg_lo,
                spread_e_neg_hi,
                spread_g_lo,
                spread_g_hi,
            )
        });

        // // Va2=Va1+Vb1+y abcd words
        meta.create_gate("s_spread_a2", |meta| {
            let s_spread_a2 = meta.query_selector(s_spread_a2);
            let spread_m_a2 = meta.query_advice(a_2, Rotation::prev());
            let spread_n_a2 = meta.query_advice(a_2, Rotation::cur());
            let spread_o_a2 = meta.query_advice(a_2, Rotation::next());
            let spread_p_a2 = meta.query_advice(a_3, Rotation::cur());

            let spread_m_a1 = meta.query_advice(a_3, Rotation::next());
            let spread_n_a1 = meta.query_advice(a_5, Rotation::cur());
            let spread_o_a1 = meta.query_advice(a_3, Rotation::prev());
            let spread_p_a1 = meta.query_advice(a_4, Rotation::cur());

            let spread_m_b1 = meta.query_advice(a_3, Rotation::next());
            let spread_n_b1 = meta.query_advice(a_5, Rotation::cur());
            let spread_o_b1 = meta.query_advice(a_3, Rotation::prev());
            let spread_p_b1 = meta.query_advice(a_4, Rotation::cur());

            let spread_m_y = meta.query_advice(a_3, Rotation::next());
            let spread_n_y = meta.query_advice(a_5, Rotation::cur());
            let spread_o_y = meta.query_advice(a_3, Rotation::prev());
            let spread_p_y = meta.query_advice(a_4, Rotation::cur());


            MixingGate::s_spread_a2(
                s_spread_a2,
                spread_m_a2,
                spread_n_a2,
                spread_o_a2,
                spread_p_a2,
                spread_m_a1,
                spread_n_a1,
                spread_o_a1,
                spread_p_a1,
                spread_m_b1,
                spread_n_b1,
                spread_o_b1,
                spread_p_b1,
                spread_m_y,
                spread_n_y,
                spread_o_y,
                spread_p_y,

            )
        });

        // s_h_prime to compute H' = H + Ch(E, F, G) + s_upper_sigma_1(E) + K + W
        meta.create_gate("s_h_prime", |meta| {
            let s_h_prime = meta.query_selector(s_h_prime);
            let h_prime_lo = meta.query_advice(a_7, Rotation::next());
            let h_prime_hi = meta.query_advice(a_8, Rotation::next());
            let h_prime_carry = meta.query_advice(a_9, Rotation::next());
            let sigma_e_lo = meta.query_advice(a_4, Rotation::cur());
            let sigma_e_hi = meta.query_advice(a_5, Rotation::cur());
            let ch_lo = meta.query_advice(a_1, Rotation::cur());
            let ch_hi = meta.query_advice(a_6, Rotation::next());
            let ch_neg_lo = meta.query_advice(a_5, Rotation::prev());
            let ch_neg_hi = meta.query_advice(a_5, Rotation::next());
            let h_lo = meta.query_advice(a_7, Rotation::prev());
            let h_hi = meta.query_advice(a_7, Rotation::cur());
            let k_lo = meta.query_advice(a_6, Rotation::prev());
            let k_hi = meta.query_advice(a_6, Rotation::cur());
            let w_lo = meta.query_advice(a_8, Rotation::prev());
            let w_hi = meta.query_advice(a_8, Rotation::cur());

            MixingGate::s_h_prime(
                s_h_prime,
                h_prime_lo,
                h_prime_hi,
                h_prime_carry,
                sigma_e_lo,
                sigma_e_hi,
                ch_lo,
                ch_hi,
                ch_neg_lo,
                ch_neg_hi,
                h_lo,
                h_hi,
                k_lo,
                k_hi,
                w_lo,
                w_hi,
            )
        });

        // Vc2=Vc1+Vd2 abcd words
        // (16, 16, 16, 16)-bit chunks
        meta.create_gate("s_spread_c2", |meta| {
            let s_spread_c2 = meta.query_selector(s_spread_c2);
            let spread_m_c2 = meta.query_advice(a_2, Rotation::prev());
            let spread_n_c2 = meta.query_advice(a_2, Rotation::cur());
            let spread_o_c2 = meta.query_advice(a_2, Rotation::next());
            let spread_p_c2 = meta.query_advice(a_3, Rotation::cur());

            let spread_m_c1 = meta.query_advice(a_3, Rotation::next());
            let spread_n_c1 = meta.query_advice(a_5, Rotation::cur());
            let spread_o_c1 = meta.query_advice(a_3, Rotation::prev());
            let spread_p_c1 = meta.query_advice(a_4, Rotation::cur());

            let spread_m_d2 = meta.query_advice(a_3, Rotation::next());
            let spread_n_d2 = meta.query_advice(a_5, Rotation::cur());
            let spread_o_d2 = meta.query_advice(a_3, Rotation::prev());
            let spread_p_d2 = meta.query_advice(a_4, Rotation::cur());


            MixingGate::s_spread_c2(
                s_spread_c2,
                spread_m_c2,
                spread_n_c2,
                spread_o_c2,
                spread_p_c2,
                spread_m_c1,
                spread_n_c1,
                spread_o_c1,
                spread_p_c1,
                spread_m_d2,
                spread_n_d2,
                spread_o_d2,
                spread_p_d2,

            )
        });

        // s_e_new
        meta.create_gate("s_e_new", |meta| {
            let s_e_new = meta.query_selector(s_e_new);
            let e_new_lo = meta.query_advice(a_8, Rotation::cur());
            let e_new_hi = meta.query_advice(a_8, Rotation::next());
            let e_new_carry = meta.query_advice(a_9, Rotation::next());
            let d_lo = meta.query_advice(a_7, Rotation::cur());
            let d_hi = meta.query_advice(a_7, Rotation::next());
            let h_prime_lo = meta.query_advice(a_7, Rotation::prev());
            let h_prime_hi = meta.query_advice(a_8, Rotation::prev());

            MixingGate::s_e_new(
                s_e_new,
                e_new_lo,
                e_new_hi,
                e_new_carry,
                d_lo,
                d_hi,
                h_prime_lo,
                h_prime_hi,
            )
        });

        // s_digest for final round
        meta.create_gate("s_digest", |meta| {
            let s_digest = meta.query_selector(s_digest);
            let lo_0 = meta.query_advice(a_3, Rotation::cur());
            let hi_0 = meta.query_advice(a_4, Rotation::cur());
            let word_0 = meta.query_advice(a_5, Rotation::cur());
            let lo_1 = meta.query_advice(a_6, Rotation::cur());
            let hi_1 = meta.query_advice(a_7, Rotation::cur());
            let word_1 = meta.query_advice(a_8, Rotation::cur());
            let lo_2 = meta.query_advice(a_3, Rotation::next());
            let hi_2 = meta.query_advice(a_4, Rotation::next());
            let word_2 = meta.query_advice(a_5, Rotation::next());
            let lo_3 = meta.query_advice(a_6, Rotation::next());
            let hi_3 = meta.query_advice(a_7, Rotation::next());
            let word_3 = meta.query_advice(a_8, Rotation::next());

            MixingGate::s_digest(
                s_digest, 
                lo_0, 
                hi_0, 
                word_0, 
                lo_1, 
                hi_1, 
                word_1, 
                lo_2, 
                hi_2, 
                word_2, 
                lo_3, 
                hi_3,
                word_3,
            )
        });

        MixingConfig {
            lookup,
            extras,
            s_vector_a1,
            s_vector_b1,
            s_vector_c1,
            s_vector_d1,
            s_vector_a2,
            s_vector_b2,
            s_vector_c2,
            s_vector_d2,
            s_decompose_abcd,
            s_decompose_efgh,
            s_decompose_ijkl,
            s_digest,
        }
    }

    /// Initialize compression with a constant Initialization Vector of 32-byte words.
    /// Returns an initialized state.
    pub(super) fn initialize_with_iv(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        init_state: [u32; STATE],
    ) -> Result<State, Error> {
        let mut new_state = State::empty_state();
        layouter.assign_region(
            || "initialize_with_iv",
            |mut region| {
                new_state = self.initialize_iv(&mut region, init_state)?;
                Ok(())
            },
        )?;
        Ok(new_state)
    }

    /// Initialize compression with some initialized state. This could be a state
    /// output from a previous compression round.
    pub(super) fn initialize_with_state(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        init_state: State,
    ) -> Result<State, Error> {
        let mut new_state = State::empty_state();
        layouter.assign_region(
            || "initialize_with_state",
            |mut region| {
                new_state = self.initialize_state(&mut region, init_state.clone())?;
                Ok(())
            },
        )?;
        Ok(new_state)
    }

    /// Given an initialized state and a message schedule, perform 64 compression rounds.
    pub(super) fn compress(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        initialized_state: State,
        w_halves: [(AssignedBits<16>, AssignedBits<16>); ROUNDS],
    ) -> Result<State, Error> {
        let mut state = State::empty_state();
        layouter.assign_region(
            || "compress",
            |mut region| {
                state = initialized_state.clone();
                for (idx, w_halves) in w_halves.iter().enumerate() {
                    state = self.assign_round(&mut region, idx.into(), state.clone(), w_halves)?;
                }
                Ok(())
            },
        )?;
        Ok(state)
    }

    /// After the final round, convert the state into the final digest.
    pub(super) fn digest(
        &self,
        layouter: &mut impl Layouter<pallas::Base>,
        state: State,
    ) -> Result<[BlockWord; DIGEST_SIZE], Error> {
        let mut digest = [BlockWord(Value::known(0)); DIGEST_SIZE];
        layouter.assign_region(
            || "digest",
            |mut region| {
                digest = self.assign_digest(&mut region, state.clone())?;

                Ok(())
            },
        )?;
        Ok(digest)
    }
}






