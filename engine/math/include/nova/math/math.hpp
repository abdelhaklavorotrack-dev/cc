#pragma once

// Minimal linear algebra for the engine. Conventions:
// - right-handed world space, +Y up, camera looks down -Z
// - Mat4 is column-major (m[column][row]); vectors are columns: v' = M * v
// - perspective() targets Vulkan clip space: depth 0..1, Y down (flip baked in)

#include "nova/core/types.hpp"

#include <array>
#include <cmath>
#include <numbers>

namespace nova::math {

inline constexpr f32 kPi = std::numbers::pi_v<f32>;
[[nodiscard]] constexpr f32 radians(f32 degrees) noexcept { return degrees * kPi / 180.0f; }

struct Vec3 {
    f32 x = 0, y = 0, z = 0;

    constexpr Vec3 operator+(const Vec3& o) const noexcept { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const noexcept { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator*(f32 s) const noexcept { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator-() const noexcept { return {-x, -y, -z}; }
    constexpr Vec3& operator+=(const Vec3& o) noexcept { return *this = *this + o; }
    constexpr Vec3& operator-=(const Vec3& o) noexcept { return *this = *this - o; }
    constexpr bool operator==(const Vec3&) const noexcept = default;
};

struct Vec4 {
    f32 x = 0, y = 0, z = 0, w = 0;
    constexpr bool operator==(const Vec4&) const noexcept = default;
};

[[nodiscard]] constexpr f32 dot(const Vec3& a, const Vec3& b) noexcept { return a.x * b.x + a.y * b.y + a.z * b.z; }
[[nodiscard]] constexpr Vec3 cross(const Vec3& a, const Vec3& b) noexcept {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
[[nodiscard]] inline f32 length(const Vec3& v) noexcept { return std::sqrt(dot(v, v)); }
[[nodiscard]] inline Vec3 normalize(const Vec3& v) noexcept {
    const f32 len = length(v);
    return len > 0.0f ? v * (1.0f / len) : Vec3{};
}

struct Mat4 {
    std::array<std::array<f32, 4>, 4> m{}; // m[column][row]

    [[nodiscard]] static constexpr Mat4 identity() noexcept {
        Mat4 r;
        for (int i = 0; i < 4; ++i) {
            r.m[i][i] = 1.0f;
        }
        return r;
    }

    [[nodiscard]] constexpr Mat4 operator*(const Mat4& b) const noexcept {
        Mat4 r;
        for (int c = 0; c < 4; ++c) {
            for (int row = 0; row < 4; ++row) {
                f32 s = 0;
                for (int k = 0; k < 4; ++k) {
                    s += m[k][row] * b.m[c][k];
                }
                r.m[c][row] = s;
            }
        }
        return r;
    }

    [[nodiscard]] constexpr Vec4 operator*(const Vec4& v) const noexcept {
        const f32 in[4] = {v.x, v.y, v.z, v.w};
        f32 out[4] = {};
        for (int row = 0; row < 4; ++row) {
            for (int c = 0; c < 4; ++c) {
                out[row] += m[c][row] * in[c];
            }
        }
        return {out[0], out[1], out[2], out[3]};
    }

    [[nodiscard]] const f32* data() const noexcept { return &m[0][0]; }
};

[[nodiscard]] constexpr Mat4 translate(const Vec3& t) noexcept {
    Mat4 r = Mat4::identity();
    r.m[3][0] = t.x;
    r.m[3][1] = t.y;
    r.m[3][2] = t.z;
    return r;
}

[[nodiscard]] constexpr Mat4 scale(const Vec3& s) noexcept {
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x;
    r.m[1][1] = s.y;
    r.m[2][2] = s.z;
    return r;
}

// Rotation by `angle` radians around normalized `axis` (right-hand rule).
[[nodiscard]] inline Mat4 rotate(const Vec3& axis, f32 angle) noexcept {
    const Vec3 a = normalize(axis);
    const f32 c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    Mat4 r = Mat4::identity();
    r.m[0][0] = t * a.x * a.x + c;
    r.m[0][1] = t * a.x * a.y + s * a.z;
    r.m[0][2] = t * a.x * a.z - s * a.y;
    r.m[1][0] = t * a.x * a.y - s * a.z;
    r.m[1][1] = t * a.y * a.y + c;
    r.m[1][2] = t * a.y * a.z + s * a.x;
    r.m[2][0] = t * a.x * a.z + s * a.y;
    r.m[2][1] = t * a.y * a.z - s * a.x;
    r.m[2][2] = t * a.z * a.z + c;
    return r;
}

[[nodiscard]] inline Mat4 look_at(const Vec3& eye, const Vec3& target, const Vec3& up) noexcept {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);
    Mat4 r = Mat4::identity();
    r.m[0][0] = s.x; r.m[1][0] = s.y; r.m[2][0] = s.z;
    r.m[0][1] = u.x; r.m[1][1] = u.y; r.m[2][1] = u.z;
    r.m[0][2] = -f.x; r.m[1][2] = -f.y; r.m[2][2] = -f.z;
    r.m[3][0] = -dot(s, eye);
    r.m[3][1] = -dot(u, eye);
    r.m[3][2] = dot(f, eye);
    return r;
}

// Vulkan clip space: x right, y down, depth in [0, 1] with near -> 0.
[[nodiscard]] inline Mat4 perspective(f32 fov_y, f32 aspect, f32 z_near, f32 z_far) noexcept {
    const f32 f = 1.0f / std::tan(fov_y * 0.5f);
    Mat4 r;
    r.m[0][0] = f / aspect;
    r.m[1][1] = -f;
    r.m[2][2] = z_far / (z_near - z_far);
    r.m[2][3] = -1.0f;
    r.m[3][2] = (z_near * z_far) / (z_near - z_far);
    return r;
}

} // namespace nova::math
