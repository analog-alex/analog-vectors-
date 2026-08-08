// @analogAlex
const std = @import("std");
const vec3 = @import("../vectors/vec3.zig");
const vec4 = @import("../vectors/vec4.zig");

/// 4x4 matrix stored in column-major order for GPU compatibility (OpenGL, Vulkan, WebGPU)
/// This is the PRIMARY transformation matrix for 3D graphics
pub const Mat4 = [16]f32;

// ===============
// Construction

pub fn from(
    m00: f32,
    m01: f32,
    m02: f32,
    m03: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    m13: f32,
    m20: f32,
    m21: f32,
    m22: f32,
    m23: f32,
    m30: f32,
    m31: f32,
    m32: f32,
    m33: f32,
) Mat4 {
    return [16]f32{
        m00, m10, m20, m30,
        m01, m11, m21, m31,
        m02, m12, m22, m32,
        m03, m13, m23, m33,
    };
}

pub fn fromCols(col0: vec4.Vec4, col1: vec4.Vec4, col2: vec4.Vec4, col3: vec4.Vec4) Mat4 {
    return [16]f32{
        col0[0], col0[1], col0[2], col0[3],
        col1[0], col1[1], col1[2], col1[3],
        col2[0], col2[1], col2[2], col2[3],
        col3[0], col3[1], col3[2], col3[3],
    };
}

pub fn identity() Mat4 {
    return [16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    };
}

pub fn zero() Mat4 {
    return [16]f32{
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
    };
}

// ===============
// Accessors

pub inline fn get(m: Mat4, row: usize, col: usize) f32 {
    return m[col * 4 + row];
}

pub inline fn getCol(m: Mat4, col: usize) vec4.Vec4 {
    const offset = col * 4;
    return vec4.Vec4{ m[offset], m[offset + 1], m[offset + 2], m[offset + 3] };
}

pub inline fn getRow(m: Mat4, row: usize) vec4.Vec4 {
    return vec4.Vec4{ m[row], m[row + 4], m[row + 8], m[row + 12] };
}

// ===============
// Matrix Operations

pub fn multiply(lhs: Mat4, rhs: Mat4) Mat4 {
    const l0: vec4.Vec4 = lhs[0..4].*;
    const l1: vec4.Vec4 = lhs[4..8].*;
    const l2: vec4.Vec4 = lhs[8..12].*;
    const l3: vec4.Vec4 = lhs[12..16].*;

    var result: Mat4 = undefined;
    inline for (0..4) |c| {
        const rc = c * 4;
        const col = vec4.mul(l0, rhs[rc]) +
            vec4.mul(l1, rhs[rc + 1]) +
            vec4.mul(l2, rhs[rc + 2]) +
            vec4.mul(l3, rhs[rc + 3]);
        result[rc..][0..4].* = col;
    }
    return result;
}

pub fn transpose(m: Mat4) Mat4 {
    return [16]f32{
        m[0], m[4], m[8],  m[12],
        m[1], m[5], m[9],  m[13],
        m[2], m[6], m[10], m[14],
        m[3], m[7], m[11], m[15],
    };
}

pub fn determinant(m: Mat4) f32 {
    const a00 = m[0];
    const a01 = m[1];
    const a02 = m[2];
    const a03 = m[3];
    const a10 = m[4];
    const a11 = m[5];
    const a12 = m[6];
    const a13 = m[7];
    const a20 = m[8];
    const a21 = m[9];
    const a22 = m[10];
    const a23 = m[11];
    const a30 = m[12];
    const a31 = m[13];
    const a32 = m[14];
    const a33 = m[15];

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    return b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
}

pub fn inverse(m: Mat4) ?Mat4 {
    const a00 = m[0];
    const a01 = m[1];
    const a02 = m[2];
    const a03 = m[3];
    const a10 = m[4];
    const a11 = m[5];
    const a12 = m[6];
    const a13 = m[7];
    const a20 = m[8];
    const a21 = m[9];
    const a22 = m[10];
    const a23 = m[11];
    const a30 = m[12];
    const a31 = m[13];
    const a32 = m[14];
    const a33 = m[15];

    const b00 = a00 * a11 - a01 * a10;
    const b01 = a00 * a12 - a02 * a10;
    const b02 = a00 * a13 - a03 * a10;
    const b03 = a01 * a12 - a02 * a11;
    const b04 = a01 * a13 - a03 * a11;
    const b05 = a02 * a13 - a03 * a12;
    const b06 = a20 * a31 - a21 * a30;
    const b07 = a20 * a32 - a22 * a30;
    const b08 = a20 * a33 - a23 * a30;
    const b09 = a21 * a32 - a22 * a31;
    const b10 = a21 * a33 - a23 * a31;
    const b11 = a22 * a33 - a23 * a32;

    const det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
    if (@abs(det) < 1e-6) return null;

    const inv_det = 1.0 / det;

    return [16]f32{
        (a11 * b11 - a12 * b10 + a13 * b09) * inv_det,
        (a02 * b10 - a01 * b11 - a03 * b09) * inv_det,
        (a31 * b05 - a32 * b04 + a33 * b03) * inv_det,
        (a22 * b04 - a21 * b05 - a23 * b03) * inv_det,
        (a12 * b08 - a10 * b11 - a13 * b07) * inv_det,
        (a00 * b11 - a02 * b08 + a03 * b07) * inv_det,
        (a32 * b02 - a30 * b05 - a33 * b01) * inv_det,
        (a20 * b05 - a22 * b02 + a23 * b01) * inv_det,
        (a10 * b10 - a11 * b08 + a13 * b06) * inv_det,
        (a01 * b08 - a00 * b10 - a03 * b06) * inv_det,
        (a30 * b04 - a31 * b02 + a33 * b00) * inv_det,
        (a21 * b02 - a20 * b04 - a23 * b00) * inv_det,
        (a11 * b07 - a10 * b09 - a12 * b06) * inv_det,
        (a00 * b09 - a01 * b07 + a02 * b06) * inv_det,
        (a31 * b01 - a30 * b03 - a32 * b00) * inv_det,
        (a20 * b03 - a21 * b01 + a22 * b00) * inv_det,
    };
}

// ===============
// Vector Transformation

pub fn transformVec3(m: Mat4, v: vec3.Vec3) vec3.Vec3 {
    const c0: vec4.Vec4 = m[0..4].*;
    const c1: vec4.Vec4 = m[4..8].*;
    const c2: vec4.Vec4 = m[8..12].*;
    const c3: vec4.Vec4 = m[12..16].*;

    const result = vec4.mul(c0, v[0]) + vec4.mul(c1, v[1]) + vec4.mul(c2, v[2]) + c3;
    return [3]f32{ result[0], result[1], result[2] };
}

pub fn transformVec4(m: Mat4, v: vec4.Vec4) vec4.Vec4 {
    const c0: vec4.Vec4 = m[0..4].*;
    const c1: vec4.Vec4 = m[4..8].*;
    const c2: vec4.Vec4 = m[8..12].*;
    const c3: vec4.Vec4 = m[12..16].*;

    return vec4.mul(c0, v[0]) + vec4.mul(c1, v[1]) + vec4.mul(c2, v[2]) + vec4.mul(c3, v[3]);
}

// ===============
// Arithmetic

pub fn add(lhs: Mat4, rhs: Mat4) Mat4 {
    var result: Mat4 = undefined;
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        result[i] = lhs[i] + rhs[i];
    }
    return result;
}

pub fn sub(lhs: Mat4, rhs: Mat4) Mat4 {
    var result: Mat4 = undefined;
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        result[i] = lhs[i] - rhs[i];
    }
    return result;
}

pub fn scale(m: Mat4, scalar: f32) Mat4 {
    var result: Mat4 = undefined;
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        result[i] = m[i] * scalar;
    }
    return result;
}

// ===============
// Transformation Constructors

pub fn translation(x: f32, y: f32, z: f32) Mat4 {
    return [16]f32{
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        x, y, z, 1,
    };
}

pub fn translationVec(v: vec3.Vec3) Mat4 {
    return translation(v[0], v[1], v[2]);
}

pub fn scaling(sx: f32, sy: f32, sz: f32) Mat4 {
    return [16]f32{
        sx, 0,  0,  0,
        0,  sy, 0,  0,
        0,  0,  sz, 0,
        0,  0,  0,  1,
    };
}

pub fn scalingVec(v: vec3.Vec3) Mat4 {
    return scaling(v[0], v[1], v[2]);
}

pub fn rotationX(radians: f32) Mat4 {
    const c = @cos(radians);
    const s = @sin(radians);
    return [16]f32{
        1, 0,  0, 0,
        0, c,  s, 0,
        0, -s, c, 0,
        0, 0,  0, 1,
    };
}

pub fn rotationY(radians: f32) Mat4 {
    const c = @cos(radians);
    const s = @sin(radians);
    return [16]f32{
        c, 0, -s, 0,
        0, 1, 0,  0,
        s, 0, c,  0,
        0, 0, 0,  1,
    };
}

pub fn rotationZ(radians: f32) Mat4 {
    const c = @cos(radians);
    const s = @sin(radians);
    return [16]f32{
        c,  s, 0, 0,
        -s, c, 0, 0,
        0,  0, 1, 0,
        0,  0, 0, 1,
    };
}

pub fn rotationAxis(axis: vec3.Vec3, radians: f32) Mat4 {
    const c = @cos(radians);
    const s = @sin(radians);
    const t = 1.0 - c;

    const n = vec3.normalize(axis);
    const x = n[0];
    const y = n[1];
    const z = n[2];

    return [16]f32{
        t * x * x + c,     t * x * y + s * z, t * x * z - s * y, 0,
        t * x * y - s * z, t * y * y + c,     t * y * z + s * x, 0,
        t * x * z + s * y, t * y * z - s * x, t * z * z + c,     0,
        0,                 0,                 0,                 1,
    };
}

pub fn perspective(fov_radians: f32, aspect: f32, near: f32, far: f32) Mat4 {
    const f = 1.0 / @tan(fov_radians / 2.0);
    const nf = 1.0 / (near - far);

    return [16]f32{
        f / aspect, 0, 0,                     0,
        0,          f, 0,                     0,
        0,          0, (far + near) * nf,     -1,
        0,          0, 2.0 * far * near * nf, 0,
    };
}

pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) Mat4 {
    const lr = 1.0 / (left - right);
    const bt = 1.0 / (bottom - top);
    const nf = 1.0 / (near - far);

    return [16]f32{
        -2.0 * lr,           0,                   0,                 0,
        0,                   -2.0 * bt,           0,                 0,
        0,                   0,                   2.0 * nf,          0,
        (left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1,
    };
}

pub fn lookAt(eye: vec3.Vec3, target: vec3.Vec3, up: vec3.Vec3) Mat4 {
    const f = vec3.normalize(vec3.sub(target, eye));
    const s = vec3.normalize(vec3.cross(f, up));
    const u = vec3.cross(s, f);

    return [16]f32{
        s[0],              u[0],              -f[0],            0,
        s[1],              u[1],              -f[1],            0,
        s[2],              u[2],              -f[2],            0,
        -vec3.dot(s, eye), -vec3.dot(u, eye), vec3.dot(f, eye), 1,
    };
}

// ===============
// Utility

pub fn equal(a: Mat4, b: Mat4) bool {
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

pub fn approxEqual(a: Mat4, b: Mat4, epsilon: f32) bool {
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        if (@abs(a[i] - b[i]) > epsilon) return false;
    }
    return true;
}

// ===============
// Tests

test "identity matrix" {
    const m = identity();
    try std.testing.expect(get(m, 0, 0) == 1);
    try std.testing.expect(get(m, 1, 1) == 1);
    try std.testing.expect(get(m, 2, 2) == 1);
    try std.testing.expect(get(m, 3, 3) == 1);
    try std.testing.expect(get(m, 0, 1) == 0);
}

test "multiply - identity matrix" {
    const m = translation(1, 2, 3);
    const id = identity();
    const result = multiply(m, id);
    try std.testing.expect(approxEqual(result, m, 0.0001));
}

test "transpose swaps rows and columns" {
    const m = from(
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    );
    const t = transpose(m);
    try std.testing.expect(get(t, 0, 1) == get(m, 1, 0));
    try std.testing.expect(get(t, 2, 3) == get(m, 3, 2));
}

test "inverse - calculates inverse matrix" {
    const m = translation(5, 10, 15);
    const inv = inverse(m).?;
    const product = multiply(m, inv);
    try std.testing.expect(approxEqual(product, identity(), 0.0001));
}

test "translation creates translation matrix" {
    const m = translation(10, 20, 30);
    const v = vec3.init(0, 0, 0);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.equal(result, vec3.init(10, 20, 30)));
}

test "scaling creates scaling matrix" {
    const m = scaling(2, 3, 4);
    const v = vec3.init(5, 6, 7);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.equal(result, vec3.init(10, 18, 28)));
}

test "rotationX rotates around x-axis" {
    const m = rotationX(std.math.pi / 2.0);
    const v = vec3.init(0, 1, 0);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.approxEqual(result, vec3.init(0, 0, 1), 0.0001));
}

test "rotationY rotates around y-axis" {
    const m = rotationY(std.math.pi / 2.0);
    const v = vec3.init(1, 0, 0);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.approxEqual(result, vec3.init(0, 0, -1), 0.0001));
}

test "rotationZ rotates around z-axis" {
    const m = rotationZ(std.math.pi / 2.0);
    const v = vec3.init(1, 0, 0);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.approxEqual(result, vec3.init(0, 1, 0), 0.0001));
}

test "rotationAxis rotates around arbitrary axis" {
    const axis = vec3.normalize(vec3.init(1, 1, 1));
    const m = rotationAxis(axis, std.math.pi * 2.0 / 3.0);
    const v = vec3.init(1, 0, 0);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.approxEqual(result, vec3.init(0, 1, 0), 0.0001));
}

test "perspective creates perspective projection matrix" {
    const m = perspective(std.math.pi / 2.0, 16.0 / 9.0, 0.1, 100.0);
    try std.testing.expect(get(m, 3, 2) == -1);
}

test "orthographic creates orthographic projection matrix" {
    const m = orthographic(-10, 10, -10, 10, 0.1, 100);
    const v = vec4.init(0, 0, 0, 1);
    const result = transformVec4(m, v);
    try std.testing.expect(@abs(result[0]) < 0.0001);
    try std.testing.expect(@abs(result[1]) < 0.0001);
}

test "lookAt creates view matrix" {
    const eye = vec3.init(0, 0, 5);
    const target = vec3.init(0, 0, 0);
    const up = vec3.init(0, 1, 0);
    const m = lookAt(eye, target, up);

    const v = vec3.init(0, 0, 0);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.approxEqual(result, vec3.init(0, 0, -5), 0.0001));
}

test "combined transformations - translate then scale" {
    const t = translation(10, 0, 0);
    const s = scaling(2, 2, 2);
    const combined = multiply(s, t);

    const v = vec3.init(0, 0, 0);
    const result = transformVec3(combined, v);
    try std.testing.expect(vec3.equal(result, vec3.init(20, 0, 0)));
}

test "transformVec3 applies translation term (point semantics)" {
    const m = translation(10, -2, 3);
    const v = vec3.init(1, 2, 3);
    const result = transformVec3(m, v);
    try std.testing.expect(vec3.equal(result, vec3.init(11, 0, 6)));
}

test "transformVec4 keeps direction unchanged by translation (w=0)" {
    const m = translation(10, -2, 3);
    const dir = vec4.init(1, 2, 3, 0);
    const result = transformVec4(m, dir);
    try std.testing.expect(vec4.approxEqual(result, dir, 0.0001));
}

test "transformVec4 translates point (w=1)" {
    const m = translation(10, -2, 3);
    const point = vec4.init(1, 2, 3, 1);
    const result = transformVec4(m, point);
    try std.testing.expect(vec4.approxEqual(result, vec4.init(11, 0, 6, 1), 0.0001));
}
