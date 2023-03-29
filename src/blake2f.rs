// implementation of blake2 hashing algorithm with halo2

// BLAKE2 Sigma constant
pub const BLAKE2B_SIGMA: [[u8; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],];

// BLAKE2 G function
fn g_function(v: &mut [u64; 16], m: &[u64; 16], r: usize, i: usize, j: usize, k: usize, l: usize) {
    v[j] = v[j].wrapping_add(v[k]).wrapping_add(m[BLAKE2B_SIGMA[r][2 * i]]);
    v[l] = (v[l] ^ v[j]).rotate_right(32);
    v[i] = v[i].wrapping_add(v[l]);
    v[k] = (v[k] ^ v[i]).rotate_right(24);
    v[j] = v[j].wrapping_add(v[k]).wrapping_add(m[BLAKE2B_SIGMA[r][2 * i + 1]]);
    v[l] = (v[l] ^ v[j]).rotate_right(16);
    v[i] = v[i].wrapping_add(v[l]);
    v[k] = (v[k] ^ v[i]).rotate_right(63);
}

