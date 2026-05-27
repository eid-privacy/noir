pub struct P256Point {
    pub x: Field,
    pub y: Field,
}

pub fn t256_ec_add(a: P256Point, b: P256Point) -> P256Point {
    _ec_add(a, b, true)
}
#[foreign(ec_add)]
fn _ec_add(_a: P256Point, _b: P256Point, _predicate: bool) -> P256Point {}


pub fn t256_msm(points: [P256Point], scalars: [Field]) -> P256Point {
    _t256_msm(points, scalars, true)
}
#[foreign(ec_msm)]
fn _t256_msm(_points: [P256Point], _scalars: [Field], _predicate: bool) -> P256Point {}

pub fn ecdsa_pop(
    public_key_x: Field,
    signature_s: [Field; 32],
    message_hash: [u8; 32],
    T: P256Point,
    U: P256Point,
) -> Field {
    _ecdsa_pop(public_key_x, signature_s, message_hash, T, U, true)
}

#[foreign(ecdsa_proof_of_possession)]
fn _ecdsa_pop(
    _public_key_x: Field,
    _signature_s: [Field; 32],
    _message_hash: [u8; 32],
    _T: P256Point,
    _U: P256Point,
    _predicate: bool,
) -> Field {}