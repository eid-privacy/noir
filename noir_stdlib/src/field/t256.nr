// Two Tom-256 scalar field elements make up a P256 point
pub struct P256Point {
    pub x: Field,
    pub y: Field,
}

pub struct T256Scalar {
    pub lo: Field,
    pub hi: Field,
}

pub fn t256_ec_add(p1: P256Point, p2: P256Point) -> P256Point {
    t256_ec_add_array_return(p1, p2, true)[0]
}

pub fn t256_msm<let N: u32>(points: [P256Point; N], scalars: [T256Scalar; N]) -> P256Point {
    t256_msm_array_return(points, scalars, true)[0]
}

#[foreign(embedded_curve_add)]
fn t256_ec_add_array_return(
    _p1: P256Point,
    _p2: P256Point,
    _predicate: bool,
) -> [P256Point; 1] {}

#[foreign(multi_scalar_mul)]
fn t256_msm_array_return<let N: u32>(
    _points: [P256Point; N],
    _scalars: [T256Scalar; N],
    _predicate: bool,
) -> [P256Point; 1] {}