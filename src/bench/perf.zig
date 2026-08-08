//! Small performance test for analog-vectors. Runs fixed-iteration benchmarks
//! for vectors, matrices, and collision (intersection) routines, and prints
//! ns/op and ops/sec metrics to stdout.
//!
//! Run with: zig build bench -Doptimize=ReleaseFast
const std = @import("std");

const vectors = @import("vectors");

const vec2 = vectors.vec2;
const vec3 = vectors.vec3;
const vec4 = vectors.vec4;
const mat2 = vectors.mat2;
const mat3 = vectors.mat3;
const mat4 = vectors.mat4;

const ray_mod = vectors.ray;
const sphere_mod = vectors.sphere;
const aabb_mod = vectors.aabb;
const plane_mod = vectors.plane;
const intersect = vectors.intersect;

const simd_ray_mod = vectors.simd_ray;
const simd_sphere_mod = vectors.simd_sphere;
const simd_intersect = vectors.simd_intersect;

const random = vectors.random;

const iterations: usize = 1_000_000;

/// Minimal monotonic-clock timer. `std.time.Timer` requires wiring up a full
/// `std.Io` implementation in this Zig version, which is unnecessary overhead
/// for a synchronous benchmark, so this reads the OS monotonic clock directly.
const Timer = struct {
    start_ns: u64,

    fn nowNs() u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    fn start() Timer {
        return .{ .start_ns = nowNs() };
    }

    fn read(self: Timer) u64 {
        return nowNs() - self.start_ns;
    }
};

const BenchResult = struct {
    name: []const u8,
    iterations: usize,
    total_ns: u64,

    fn nsPerOp(self: BenchResult) f64 {
        return @as(f64, @floatFromInt(self.total_ns)) / @as(f64, @floatFromInt(self.iterations));
    }

    fn opsPerSec(self: BenchResult) f64 {
        return 1_000_000_000.0 / self.nsPerOp();
    }
};

fn printResult(r: BenchResult) void {
    std.debug.print("  {s:<28} {d:>12.2} ns/op  {d:>16.0} ops/sec\n", .{
        r.name, r.nsPerOp(), r.opsPerSec(),
    });
}

// ===============
// Vector benchmarks

fn benchVec3Sum() BenchResult {
    var a = vec3.init(1.0, 2.0, 3.0);
    var b = vec3.init(4.0, 5.0, 6.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = vec3.sum(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "vec3.sum", .iterations = iterations, .total_ns = elapsed };
}

fn benchVec3Dot() BenchResult {
    var a = vec3.init(1.0, 2.0, 3.0);
    var b = vec3.init(4.0, 5.0, 6.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = vec3.dot(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "vec3.dot", .iterations = iterations, .total_ns = elapsed };
}

fn benchVec3Cross() BenchResult {
    var a = vec3.init(1.0, 2.0, 3.0);
    var b = vec3.init(4.0, 5.0, 6.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = vec3.cross(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "vec3.cross", .iterations = iterations, .total_ns = elapsed };
}

fn benchVec3Normalize() BenchResult {
    var a = vec3.init(3.0, 4.0, 12.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        const result = vec3.normalize(a);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "vec3.normalize", .iterations = iterations, .total_ns = elapsed };
}

fn benchVec4Sum() BenchResult {
    var a = vec4.init(1.0, 2.0, 3.0, 4.0);
    var b = vec4.init(5.0, 6.0, 7.0, 8.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = vec4.sum(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "vec4.sum (simd)", .iterations = iterations, .total_ns = elapsed };
}

fn benchVectors() void {
    printResult(benchVec3Sum());
    printResult(benchVec3Dot());
    printResult(benchVec3Cross());
    printResult(benchVec3Normalize());
    printResult(benchVec4Sum());
}

// ===============
// Matrix benchmarks

fn benchMat2Multiply() BenchResult {
    var a = mat2.identity();
    var b = mat2.rotation(0.5);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = mat2.multiply(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "mat2.multiply", .iterations = iterations, .total_ns = elapsed };
}

fn benchMat3Multiply() BenchResult {
    var a = mat3.identity();
    var b = mat3.identity();

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = mat3.multiply(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "mat3.multiply", .iterations = iterations, .total_ns = elapsed };
}

fn benchMat4Multiply() BenchResult {
    var a = mat4.rotationX(0.3);
    var b = mat4.translation(1.0, 2.0, 3.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&a);
        std.mem.doNotOptimizeAway(&b);
        const result = mat4.multiply(a, b);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "mat4.multiply", .iterations = iterations, .total_ns = elapsed };
}

fn benchMat4Inverse() BenchResult {
    var m = mat4.multiply(mat4.rotationY(0.7), mat4.translation(2.0, -1.0, 5.0));

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&m);
        const result = mat4.inverse(m);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "mat4.inverse", .iterations = iterations, .total_ns = elapsed };
}

fn benchMat4TransformVec3() BenchResult {
    var m = mat4.rotationZ(0.4);
    var v = vec3.init(1.0, 2.0, 3.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&m);
        std.mem.doNotOptimizeAway(&v);
        const result = mat4.transformVec3(m, v);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "mat4.transformVec3", .iterations = iterations, .total_ns = elapsed };
}

fn benchMatrices() void {
    printResult(benchMat2Multiply());
    printResult(benchMat3Multiply());
    printResult(benchMat4Multiply());
    printResult(benchMat4Inverse());
    printResult(benchMat4TransformVec3());
}

// ===============
// Collision benchmarks

fn benchRaySphere() BenchResult {
    var r = ray_mod.from(vec3.init(-10.0, 0.0, 0.0), vec3.init(1.0, 0.0, 0.0));
    var s = sphere_mod.from(vec3.init(0.0, 0.0, 0.0), 3.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&r);
        std.mem.doNotOptimizeAway(&s);
        const result = intersect.raySphere(r, s);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "intersect.raySphere", .iterations = iterations, .total_ns = elapsed };
}

fn benchRayAABB() BenchResult {
    var r = ray_mod.from(vec3.init(-10.0, 0.0, 0.0), vec3.init(1.0, 0.0, 0.0));
    var box = aabb_mod.from(vec3.init(-1.0, -1.0, -1.0), vec3.init(1.0, 1.0, 1.0));

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&r);
        std.mem.doNotOptimizeAway(&box);
        const result = intersect.rayAABB(r, box);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "intersect.rayAABB", .iterations = iterations, .total_ns = elapsed };
}

fn benchRayPlane() BenchResult {
    var r = ray_mod.from(vec3.init(0.0, 5.0, 0.0), vec3.init(0.0, -1.0, 0.0));
    var p = plane_mod.fromPointNormal(vec3.init(0.0, 0.0, 0.0), vec3.init(0.0, 1.0, 0.0));

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&r);
        std.mem.doNotOptimizeAway(&p);
        const result = intersect.rayPlane(r, p);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "intersect.rayPlane", .iterations = iterations, .total_ns = elapsed };
}

fn benchSimdRaySphere() BenchResult {
    var r = simd_ray_mod.from(vec4.init(-10.0, 0.0, 0.0, 1.0), vec4.init(1.0, 0.0, 0.0, 0.0));
    var s = simd_sphere_mod.from(vec4.init(0.0, 0.0, 0.0, 1.0), 3.0);

    var timer = Timer.start();
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        std.mem.doNotOptimizeAway(&r);
        std.mem.doNotOptimizeAway(&s);
        const result = simd_intersect.raySphere(r, s);
        std.mem.doNotOptimizeAway(&result);
    }
    const elapsed = timer.read();
    return .{ .name = "simd_intersect.raySphere", .iterations = iterations, .total_ns = elapsed };
}

/// Scene-scale benchmark: cast a batch of rays against a batch of spheres
/// (broad-phase style N-vs-N workload) and report throughput plus hit count.
fn benchSceneRaysVsSpheres() BenchResult {
    const scene_size = 512;
    var prng = std.Random.DefaultPrng.init(0xA10E_5EED);
    const rng = prng.random();

    var rays: [scene_size]ray_mod.Ray = undefined;
    var spheres: [scene_size]sphere_mod.Sphere = undefined;
    for (0..scene_size) |idx| {
        const origin = vec3.fromArray(random.randomPointInUnitSphere(rng));
        const dir = vec3.fromArray(random.randomDirection3D(rng));
        rays[idx] = ray_mod.from(vec3.mul(origin, 20.0), dir);
        spheres[idx] = sphere_mod.from(vec3.fromArray(random.randomPointInUnitSphere(rng)), random.randomFloat(rng, 0.1, 2.0));
    }

    var hits: usize = 0;
    var timer = Timer.start();
    for (rays) |r| {
        for (spheres) |s| {
            if (intersect.raySphere(r, s)) |_| hits += 1;
        }
        std.mem.doNotOptimizeAway(&hits);
    }
    const elapsed = timer.read();

    const total_checks = scene_size * scene_size;
    std.debug.print("  ({d} rays x {d} spheres = {d} checks, {d} hits)\n", .{ scene_size, scene_size, total_checks, hits });
    return .{ .name = "scene raySphere batch", .iterations = total_checks, .total_ns = elapsed };
}

fn benchCollisions() void {
    printResult(benchRaySphere());
    printResult(benchRayAABB());
    printResult(benchRayPlane());
    printResult(benchSimdRaySphere());
    printResult(benchSceneRaysVsSpheres());
}

// ===============
// Entry point

pub fn main() !void {
    std.debug.print("=== analog-vectors performance metrics ===\n", .{});
    std.debug.print("iterations per micro-benchmark: {d}\n\n", .{iterations});

    std.debug.print("-- Vectors --\n", .{});
    benchVectors();

    std.debug.print("\n-- Matrices --\n", .{});
    benchMatrices();

    std.debug.print("\n-- Collisions --\n", .{});
    benchCollisions();
}
