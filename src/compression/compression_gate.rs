use super::utils::*;

pub struct CompressionGate<F: Field>(PhantomData<F>);

impl<F: PrimeField> CompressionGate<F> {
    fn ones() -> Expression<F> {
        Expression::Constant(F::ONE)
    }

    // Implement G function
    pub fn g_func(Vec<StateChunk>>, a: Value<F>, b: Value<F>, c: Value<F>, d: Value<F>, x: MessageChunk, y: MessageChunk) -> Vec<StateChunk>> {
        let w = 64; // Word size
        let r1 = 32;
        let r2 = 24;
        let r3 = 16;
        let r4 = 64;
    
        let tmp1 = v[a] + v[b] + x;
        let tmp2 = v[d] ^ tmp1;
        let tmp3 = v[c] + tmp2;
        let tmp4 = v[b] ^ tmp3;
        let tmp5 = v[a] + tmp4 + y;
        let tmp6 = v[d] ^ tmp5;
        let tmp7 = v[c] + tmp6;
        let tmp8 = v[b] ^ tmp7;
    
        v[a] = tmp1;
        v[d] = tmp2.rotate_right(r1);
        v[c] = tmp3;
        v[b] = tmp4.rotate_right(r2);
        v[a] = tmp5;
        v[d] = tmp6.rotate_right(r3);
        v[c] = tmp7;
        v[b] = tmp8.rotate_right(r4);
    
        v.iter().cloned().collect()

    }
}