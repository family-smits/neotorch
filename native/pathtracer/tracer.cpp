// NeoTorch Ultra Path Tracer — GGX PBR · stratified · HDRF out (C++17)
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Vec3 {
    double x = 0, y = 0, z = 0;
    Vec3() = default;
    Vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(const Vec3& o) const { return {x * o.x, y * o.y, z * o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(double s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double length() const { return std::sqrt(dot(*this)); }
    Vec3 normalized() const {
        double L = length();
        return L > 1e-12 ? (*this) / L : Vec3(0, 1, 0);
    }
    double luminance() const { return 0.2126 * x + 0.7152 * y + 0.0722 * z; }
    Vec3 clamp01() const {
        return {std::clamp(x, 0.0, 1.0), std::clamp(y, 0.0, 1.0), std::clamp(z, 0.0, 1.0)};
    }
};
inline Vec3 operator*(double s, const Vec3& v) { return v * s; }

struct Ray {
    Vec3 o, d;
    Ray(const Vec3& o, const Vec3& d) : o(o), d(d.normalized()) {}
    Vec3 at(double t) const { return o + d * t; }
};

struct Hit {
    double t = 1e30;
    Vec3 p, n, albedo;
    double roughness = 0.45, metal = 0.05;
    bool hit = false;
    bool is_floor = false;
};

struct Tri {
    Vec3 v0, v1, v2;
    Vec3 n0, n1, n2;
    Vec3 c0, c1, c2;
    Vec3 centroid;
    Vec3 bmin, bmax;
    bool floor = false;
};

struct AABB {
    Vec3 bmin{1e30, 1e30, 1e30}, bmax{-1e30, -1e30, -1e30};
    void expand(const Vec3& p) {
        bmin.x = std::min(bmin.x, p.x); bmin.y = std::min(bmin.y, p.y); bmin.z = std::min(bmin.z, p.z);
        bmax.x = std::max(bmax.x, p.x); bmax.y = std::max(bmax.y, p.y); bmax.z = std::max(bmax.z, p.z);
    }
    bool hit(const Ray& r, double tmin, double tmax) const {
        for (int a = 0; a < 3; ++a) {
            double orig = (&r.o.x)[a], dir = (&r.d.x)[a];
            double mn = (&bmin.x)[a], mx = (&bmax.x)[a];
            double invD = 1.0 / (std::fabs(dir) < 1e-15 ? (dir >= 0 ? 1e-15 : -1e-15) : dir);
            double t0 = (mn - orig) * invD;
            double t1 = (mx - orig) * invD;
            if (invD < 0) std::swap(t0, t1);
            tmin = t0 > tmin ? t0 : tmin;
            tmax = t1 < tmax ? t1 : tmax;
            if (tmax <= tmin) return false;
        }
        return true;
    }
};

struct RNG {
    uint64_t s;
    explicit RNG(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint32_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return (uint32_t)((s * 0x2545F4914F6CDD1DULL) >> 32);
    }
    double uniform() { return (next() >> 8) * (1.0 / 16777216.0); }
};

static void orthonormal(const Vec3& n, Vec3& t, Vec3& b) {
    Vec3 w = n.normalized();
    t = (std::fabs(w.x) > 0.1 ? Vec3(0, 1, 0) : Vec3(1, 0, 0)).cross(w).normalized();
    b = w.cross(t);
}

Vec3 cosine_hemisphere(RNG& rng, const Vec3& n) {
    double r1 = rng.uniform(), r2 = rng.uniform();
    double phi = 2 * M_PI * r1;
    double su = std::sqrt(r2), cu = std::sqrt(std::max(0.0, 1 - r2));
    Vec3 t, b; orthonormal(n, t, b);
    return (t * std::cos(phi) * su + b * std::sin(phi) * su + n.normalized() * cu).normalized();
}

// GGX / Trowbridge-Reitz
double ggx_D(double NoH, double a) {
    double a2 = a * a;
    double d = NoH * NoH * (a2 - 1.0) + 1.0;
    return a2 / (M_PI * d * d + 1e-12);
}
double ggx_G1(double NoV, double a) {
    double k = a * a / 2.0;
    return NoV / (NoV * (1.0 - k) + k + 1e-12);
}
double ggx_G(double NoV, double NoL, double a) {
    return ggx_G1(NoV, a) * ggx_G1(NoL, a);
}
Vec3 fresnel_schlick(double cosTheta, const Vec3& F0) {
    double t = std::pow(1.0 - std::clamp(cosTheta, 0.0, 1.0), 5.0);
    return F0 + (Vec3(1, 1, 1) - F0) * t;
}
Vec3 sample_ggx(RNG& rng, const Vec3& n, double roughness) {
    double a = std::max(0.001, roughness * roughness);
    double u1 = rng.uniform(), u2 = rng.uniform();
    double phi = 2 * M_PI * u1;
    double cosTheta = std::sqrt((1.0 - u2) / (1.0 + (a * a - 1.0) * u2 + 1e-12));
    double sinTheta = std::sqrt(std::max(0.0, 1.0 - cosTheta * cosTheta));
    Vec3 h_local(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, cosTheta);
    Vec3 t, b; orthonormal(n, t, b);
    return (t * h_local.x + b * h_local.y + n.normalized() * h_local.z).normalized();
}

bool intersect_tri(const Tri& t, const Ray& r, double tmin, double tmax, Hit& h) {
    Vec3 e1 = t.v1 - t.v0, e2 = t.v2 - t.v0;
    Vec3 pvec = r.d.cross(e2);
    double det = e1.dot(pvec);
    if (std::fabs(det) < 1e-12) return false;
    double inv = 1.0 / det;
    Vec3 tvec = r.o - t.v0;
    double u = tvec.dot(pvec) * inv;
    if (u < 0 || u > 1) return false;
    Vec3 qvec = tvec.cross(e1);
    double v = r.d.dot(qvec) * inv;
    if (v < 0 || u + v > 1) return false;
    double tt = e2.dot(qvec) * inv;
    if (tt <= tmin || tt >= tmax) return false;
    double w = 1 - u - v;
    h.hit = true;
    h.t = tt;
    h.p = r.at(tt);
    h.n = (t.n0 * w + t.n1 * u + t.n2 * v).normalized();
    if (h.n.dot(r.d) > 0) h.n = -h.n;
    h.albedo = (t.c0 * w + t.c1 * u + t.c2 * v).clamp01();
    h.is_floor = t.floor;
    // Derive PBR from albedo (photoreal material guess)
    double lum = h.albedo.luminance();
    double sat = std::max({h.albedo.x, h.albedo.y, h.albedo.z}) - std::min({h.albedo.x, h.albedo.y, h.albedo.z});
    if (t.floor) {
        h.roughness = 0.72;
        h.metal = 0.0;
    } else {
        h.metal = std::clamp(0.02 + (1.0 - sat) * 0.55 * (lum > 0.55 ? 1.0 : 0.25), 0.0, 0.92);
        h.roughness = std::clamp(0.18 + sat * 0.55 + (1.0 - lum) * 0.22 - h.metal * 0.12, 0.045, 0.92);
    }
    return true;
}

struct BVHNode {
    AABB box;
    int left = -1, right = -1;
    int start = 0, count = 0;
};

struct Scene {
    std::vector<Tri> tris;
    std::vector<BVHNode> nodes;
    AABB world;
    AABB subject;

    void build() {
        nodes.clear();
        if (tris.empty()) return;
        std::vector<int> idx(tris.size());
        for (size_t i = 0; i < tris.size(); ++i) idx[i] = (int)i;
        std::vector<Tri> flat;
        flat.reserve(tris.size());
        std::vector<BVHNode> rebuilt;
        std::function<int(int, int)> rebuild = [&](int start, int end) -> int {
            AABB box;
            for (int i = start; i < end; ++i) {
                box.expand(tris[idx[i]].bmin);
                box.expand(tris[idx[i]].bmax);
            }
            BVHNode node;
            node.box = box;
            node.start = (int)flat.size();
            int n = end - start;
            int id = (int)rebuilt.size();
            rebuilt.push_back(node);
            if (n <= 4) {
                for (int i = start; i < end; ++i) flat.push_back(tris[idx[i]]);
                rebuilt[id].count = n;
                return id;
            }
            Vec3 ext = box.bmax - box.bmin;
            int axis = 0;
            if (ext.y > ext.x) axis = 1;
            if (ext.z > (&ext.x)[axis]) axis = 2;
            int mid = start + n / 2;
            std::nth_element(idx.begin() + start, idx.begin() + mid, idx.begin() + end,
                [&](int a, int b) {
                    return (&tris[a].centroid.x)[axis] < (&tris[b].centroid.x)[axis];
                });
            int L = rebuild(start, mid);
            int R = rebuild(mid, end);
            rebuilt[id].left = L;
            rebuilt[id].right = R;
            rebuilt[id].count = 0;
            return id;
        };
        rebuild(0, (int)idx.size());
        tris.swap(flat);
        nodes.swap(rebuilt);
        world = nodes.empty() ? AABB{} : nodes[0].box;
        subject = AABB{};
        bool any = false;
        for (const auto& t : tris) {
            if (t.floor) continue;
            subject.expand(t.bmin);
            subject.expand(t.bmax);
            any = true;
        }
        if (!any) subject = world;
    }

    bool hit(const Ray& r, Hit& out) const {
        if (nodes.empty()) return false;
        out.hit = false;
        out.t = 1e30;
        int stack[64];
        int sp = 0;
        stack[sp++] = 0;
        while (sp) {
            int ni = stack[--sp];
            const BVHNode& n = nodes[ni];
            if (!n.box.hit(r, 1e-4, out.t)) continue;
            if (n.count > 0) {
                for (int i = 0; i < n.count; ++i) {
                    Hit h;
                    if (intersect_tri(tris[n.start + i], r, 1e-4, out.t, h)) out = h;
                }
            } else {
                if (n.left >= 0) stack[sp++] = n.left;
                if (n.right >= 0) stack[sp++] = n.right;
            }
        }
        return out.hit;
    }
};

static void add_floor(Scene& scene) {
    AABB wb;
    for (const auto& t : scene.tris) { wb.expand(t.bmin); wb.expand(t.bmax); }
    double y0 = wb.bmin.y - 0.001;
    double cx = 0.5 * (wb.bmin.x + wb.bmax.x);
    double cz = 0.5 * (wb.bmin.z + wb.bmax.z);
    double span = std::max(wb.bmax.x - wb.bmin.x, wb.bmax.z - wb.bmin.z) * 3.2;
    if (span < 1.5) span = 2.5;
    Vec3 fl(0.62, 0.62, 0.64);
    Vec3 p0(cx - span, y0, cz - span), p1(cx + span, y0, cz - span);
    Vec3 p2(cx + span, y0, cz + span), p3(cx - span, y0, cz + span);
    auto add = [&](Vec3 a, Vec3 b, Vec3 c) {
        Tri t;
        t.v0 = a; t.v1 = b; t.v2 = c;
        t.n0 = t.n1 = t.n2 = Vec3(0, 1, 0);
        t.c0 = t.c1 = t.c2 = fl;
        t.centroid = (a + b + c) / 3.0;
        t.bmin.x = std::min({a.x, b.x, c.x}); t.bmin.y = y0; t.bmin.z = std::min({a.z, b.z, c.z});
        t.bmax.x = std::max({a.x, b.x, c.x}); t.bmax.y = y0; t.bmax.z = std::max({a.z, b.z, c.z});
        t.floor = true;
        scene.tris.push_back(t);
    };
    add(p0, p1, p2);
    add(p0, p2, p3);
}

bool load_ntmesh(const std::string& path, Scene& scene) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    char mag[4];
    in.read(mag, 4);
    if (std::string(mag, 4) != "NTM1") return false;
    uint32_t nv = 0, nf = 0;
    in.read(reinterpret_cast<char*>(&nv), 4);
    in.read(reinterpret_cast<char*>(&nf), 4);
    if (!nv || !nf) return false;
    std::vector<float> pos(nv * 3), col(nv * 3), nrm(nv * 3);
    std::vector<int32_t> idx(nf * 3);
    in.read(reinterpret_cast<char*>(pos.data()), nv * 3 * 4);
    in.read(reinterpret_cast<char*>(col.data()), nv * 3 * 4);
    in.read(reinterpret_cast<char*>(nrm.data()), nv * 3 * 4);
    in.read(reinterpret_cast<char*>(idx.data()), nf * 3 * 4);
    if (!in) return false;
    for (uint32_t i = 0; i < nf; ++i) {
        int a = idx[i * 3], b = idx[i * 3 + 1], c = idx[i * 3 + 2];
        if (a < 0 || b < 0 || c < 0 || (uint32_t)a >= nv || (uint32_t)b >= nv || (uint32_t)c >= nv) continue;
        Tri t;
        t.v0 = {pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2]};
        t.v1 = {pos[b * 3], pos[b * 3 + 1], pos[b * 3 + 2]};
        t.v2 = {pos[c * 3], pos[c * 3 + 1], pos[c * 3 + 2]};
        t.c0 = Vec3(col[a * 3], col[a * 3 + 1], col[a * 3 + 2]).clamp01();
        t.c1 = Vec3(col[b * 3], col[b * 3 + 1], col[b * 3 + 2]).clamp01();
        t.c2 = Vec3(col[c * 3], col[c * 3 + 1], col[c * 3 + 2]).clamp01();
        t.n0 = Vec3(nrm[a * 3], nrm[a * 3 + 1], nrm[a * 3 + 2]).normalized();
        t.n1 = Vec3(nrm[b * 3], nrm[b * 3 + 1], nrm[b * 3 + 2]).normalized();
        t.n2 = Vec3(nrm[c * 3], nrm[c * 3 + 1], nrm[c * 3 + 2]).normalized();
        t.centroid = (t.v0 + t.v1 + t.v2) / 3.0;
        t.bmin.x = std::min({t.v0.x, t.v1.x, t.v2.x});
        t.bmin.y = std::min({t.v0.y, t.v1.y, t.v2.y});
        t.bmin.z = std::min({t.v0.z, t.v1.z, t.v2.z});
        t.bmax.x = std::max({t.v0.x, t.v1.x, t.v2.x});
        t.bmax.y = std::max({t.v0.y, t.v1.y, t.v2.y});
        t.bmax.z = std::max({t.v0.z, t.v1.z, t.v2.z});
        scene.tris.push_back(t);
    }
    add_floor(scene);
    return !scene.tris.empty();
}

bool load_obj(const std::string& path, Scene& scene) {
    std::ifstream in(path);
    if (!in) return false;
    std::vector<Vec3> verts, colors;
    struct FaceIdx { int a, b, c; };
    std::vector<FaceIdx> faces;
    std::string line;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            double x, y, z, r = 0.72, g = 0.72, b = 0.74;
            ss >> x >> y >> z;
            if (!(ss >> r >> g >> b)) { r = 0.72; g = 0.72; b = 0.74; }
            verts.emplace_back(x, y, z);
            colors.emplace_back(Vec3(r, g, b).clamp01());
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream ss(line.substr(2));
            std::vector<int> idx;
            std::string tok;
            while (ss >> tok) {
                int vi = std::atoi(tok.c_str());
                if (vi < 0) vi = (int)verts.size() + vi + 1;
                idx.push_back(vi - 1);
            }
            for (size_t i = 1; i + 1 < idx.size(); ++i) {
                int ia = idx[0], ib = idx[i], ic = idx[i + 1];
                if (ia < 0 || ib < 0 || ic < 0 ||
                    ia >= (int)verts.size() || ib >= (int)verts.size() || ic >= (int)verts.size())
                    continue;
                faces.push_back({ia, ib, ic});
            }
        }
    }
    if (faces.empty()) return false;
    std::vector<Vec3> vnorm(verts.size(), Vec3(0, 0, 0));
    for (const auto& f : faces) {
        Vec3 fn = (verts[f.b] - verts[f.a]).cross(verts[f.c] - verts[f.a]);
        vnorm[f.a] = vnorm[f.a] + fn;
        vnorm[f.b] = vnorm[f.b] + fn;
        vnorm[f.c] = vnorm[f.c] + fn;
    }
    for (auto& n : vnorm) n = n.normalized();
    for (const auto& f : faces) {
        Tri t;
        t.v0 = verts[f.a]; t.v1 = verts[f.b]; t.v2 = verts[f.c];
        t.c0 = colors[f.a]; t.c1 = colors[f.b]; t.c2 = colors[f.c];
        t.n0 = vnorm[f.a]; t.n1 = vnorm[f.b]; t.n2 = vnorm[f.c];
        t.centroid = (t.v0 + t.v1 + t.v2) / 3.0;
        t.bmin.x = std::min({t.v0.x, t.v1.x, t.v2.x});
        t.bmin.y = std::min({t.v0.y, t.v1.y, t.v2.y});
        t.bmin.z = std::min({t.v0.z, t.v1.z, t.v2.z});
        t.bmax.x = std::max({t.v0.x, t.v1.x, t.v2.x});
        t.bmax.y = std::max({t.v0.y, t.v1.y, t.v2.y});
        t.bmax.z = std::max({t.v0.z, t.v1.z, t.v2.z});
        scene.tris.push_back(t);
    }
    add_floor(scene);
    return true;
}

Vec3 env_map(const Vec3& d) {
    double elev = d.y;
    double t = std::clamp(0.5 * (elev + 1.0), 0.0, 1.0);
    Vec3 ground(0.42, 0.40, 0.38);
    Vec3 horizon(0.78, 0.82, 0.90);
    Vec3 zenith(0.18, 0.32, 0.62);
    Vec3 col = elev < 0 ? (ground * (1 + elev) + horizon * (-elev))
                        : (horizon * (1 - t) + zenith * t);
    Vec3 sun_dir = Vec3(0.28, 0.86, 0.32).normalized();
    double sun = std::pow(std::max(0.0, d.dot(sun_dir)), 512.0);
    col = col + Vec3(12.0, 10.5, 8.0) * sun;
    // soft sun halo
    double halo = std::pow(std::max(0.0, d.dot(sun_dir)), 32.0);
    col = col + Vec3(1.8, 1.4, 0.9) * halo;
    Vec3 fill = Vec3(-0.55, 0.35, -0.4).normalized();
    col = col + Vec3(0.45, 0.38, 0.32) * std::max(0.0, d.dot(fill)) * 0.55;
    return col;
}

// Rectangular area key light above subject
struct AreaLight {
    Vec3 center, u, v, emit;
    double area = 1.0;
};

Vec3 sample_area_light(const AreaLight& L, RNG& rng, const Vec3& p, Vec3& wi, double& pdf) {
    Vec3 lp = L.center + L.u * (rng.uniform() * 2 - 1) + L.v * (rng.uniform() * 2 - 1);
    Vec3 to = lp - p;
    double dist2 = to.dot(to);
    wi = to.normalized();
    double cosA = std::fabs(L.u.cross(L.v).normalized().dot(-wi));
    pdf = dist2 / (std::max(1e-8, cosA) * L.area + 1e-12);
    return L.emit;
}

Vec3 trace(const Scene& scene, const AreaLight& light, Ray ray, RNG& rng, int max_depth) {
    Vec3 throughput(1, 1, 1);
    Vec3 color(0, 0, 0);
    for (int depth = 0; depth < max_depth; ++depth) {
        Hit h;
        if (!scene.hit(ray, h)) {
            color = color + throughput * env_map(ray.d);
            break;
        }
        Vec3 N = h.n;
        Vec3 V = -ray.d;
        Vec3 albedo = h.albedo;
        double rough = h.roughness;
        double metal = h.metal;
        Vec3 F0 = Vec3(0.04, 0.04, 0.04) * (1.0 - metal) + albedo * metal;
        Vec3 kd = albedo * (1.0 - metal);

        // Next-event: area light
        {
            Vec3 wi; double pdf;
            Vec3 Le = sample_area_light(light, rng, h.p, wi, pdf);
            double NoL = N.dot(wi);
            if (NoL > 0 && pdf > 0) {
                Hit sh;
                Ray shadow(h.p + N * 2e-4, wi);
                double dist = (light.center - h.p).length() * 1.8;
                bool occluded = scene.hit(shadow, sh) && sh.t < dist;
                if (!occluded) {
                    Vec3 H = (V + wi).normalized();
                    double NoV = std::max(1e-4, N.dot(V));
                    double NoH = std::max(0.0, N.dot(H));
                    double VoH = std::max(0.0, V.dot(H));
                    double a = std::max(0.001, rough * rough);
                    Vec3 F = fresnel_schlick(VoH, F0);
                    double D = ggx_D(NoH, a);
                    double G = ggx_G(NoV, NoL, a);
                    Vec3 spec = F * (D * G / (4.0 * NoV * NoL + 1e-12));
                    Vec3 brdf = kd / M_PI + spec;
                    color = color + throughput * brdf * Le * NoL / std::max(pdf, 1e-8);
                }
            }
        }
        // Env next-event (sun direction cone)
        {
            Vec3 sun_dir = Vec3(0.28, 0.86, 0.32).normalized();
            Vec3 wi = (sun_dir + cosine_hemisphere(rng, sun_dir) * 0.025).normalized();
            double NoL = N.dot(wi);
            if (NoL > 0) {
                Hit sh;
                if (!scene.hit(Ray(h.p + N * 2e-4, wi), sh)) {
                    Vec3 H = (V + wi).normalized();
                    double NoV = std::max(1e-4, N.dot(V));
                    double NoH = std::max(0.0, N.dot(H));
                    double VoH = std::max(0.0, V.dot(H));
                    double a = std::max(0.001, rough * rough);
                    Vec3 F = fresnel_schlick(VoH, F0);
                    Vec3 spec = F * (ggx_D(NoH, a) * ggx_G(NoV, NoL, a) / (4.0 * NoV * NoL + 1e-12));
                    color = color + throughput * (kd / M_PI + spec) * env_map(wi) * NoL * 0.65;
                }
            }
        }

        // Indirect bounce: mix diffuse / specular by Fresnel
        Vec3 F_at_n = fresnel_schlick(std::max(0.0, N.dot(V)), F0);
        double spec_w = std::clamp(F_at_n.luminance() * 0.85 + metal * 0.5, 0.05, 0.95);
        Vec3 wi;
        Vec3 weight;
        if (rng.uniform() < spec_w) {
            Vec3 H = sample_ggx(rng, N, rough);
            wi = (ray.d - H * (2.0 * ray.d.dot(H))).normalized(); // reflect - wait use V
            wi = (2.0 * V.dot(H) * H - V).normalized();
            if (wi.dot(N) <= 0) break;
            double NoL = N.dot(wi);
            double NoV = std::max(1e-4, N.dot(V));
            double NoH = std::max(1e-4, N.dot(H));
            double VoH = std::max(1e-4, V.dot(H));
            double a = std::max(0.001, rough * rough);
            Vec3 F = fresnel_schlick(VoH, F0);
            double G = ggx_G(NoV, NoL, a);
            // GGX PDF cancel with D roughly → F*G*VoH/(NoV*NoH)
            weight = F * (G * VoH / (NoV * NoH + 1e-12)) / spec_w;
        } else {
            wi = cosine_hemisphere(rng, N);
            weight = kd / (1.0 - spec_w);
        }
        throughput = throughput * weight;
        if (throughput.luminance() < 1e-5) break;
        ray = Ray(h.p + N * 2e-4, wi);
        if (depth > 2) {
            double p = std::clamp(throughput.luminance(), 0.05, 0.95);
            if (rng.uniform() > p) break;
            throughput = throughput / p;
        }
    }
    return color;
}

Vec3 aces(Vec3 x) {
    const double a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    auto tone = [&](double v) {
        return std::clamp((v * (a * v + b)) / (v * (c * v + d) + e), 0.0, 1.0);
    };
    return {tone(x.x), tone(x.y), tone(x.z)};
}

bool write_bmp(const std::string& path, int w, int h, const std::vector<Vec3>& img) {
    int row = (w * 3 + 3) & ~3;
    int data = row * h;
    std::vector<uint8_t> buf(14 + 40 + data, 0);
    auto put32 = [&](int o, uint32_t v) {
        buf[o] = v & 255; buf[o + 1] = (v >> 8) & 255; buf[o + 2] = (v >> 16) & 255; buf[o + 3] = (v >> 24) & 255;
    };
    buf[0] = 'B'; buf[1] = 'M';
    put32(2, (uint32_t)buf.size());
    put32(10, 54);
    put32(14, 40);
    put32(18, (uint32_t)w);
    put32(22, (uint32_t)h);
    buf[26] = 1; buf[28] = 24;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            Vec3 c = aces(img[y * w + x] * 0.85);
            c = {std::pow(c.x, 1.0 / 2.2), std::pow(c.y, 1.0 / 2.2), std::pow(c.z, 1.0 / 2.2)};
            int o = 54 + y * row + x * 3;
            buf[o + 0] = (uint8_t)std::clamp(int(c.z * 255.0 + 0.5), 0, 255);
            buf[o + 1] = (uint8_t)std::clamp(int(c.y * 255.0 + 0.5), 0, 255);
            buf[o + 2] = (uint8_t)std::clamp(int(c.x * 255.0 + 0.5), 0, 255);
        }
    }
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(buf.data()), (std::streamsize)buf.size());
    return true;
}

bool write_hdrf(const std::string& path, int w, int h, const std::vector<Vec3>& img) {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write("HDRF", 4);
    uint32_t uw = (uint32_t)w, uh = (uint32_t)h;
    out.write(reinterpret_cast<const char*>(&uw), 4);
    out.write(reinterpret_cast<const char*>(&uh), 4);
    std::vector<float> buf(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        buf[i * 3 + 0] = (float)img[i].x;
        buf[i * 3 + 1] = (float)img[i].y;
        buf[i * 3 + 2] = (float)img[i].z;
    }
    out.write(reinterpret_cast<const char*>(buf.data()), (std::streamsize)buf.size() * sizeof(float));
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: pathtracer <in.obj|.ntmesh> <out.bmp> [w] [h] [spp] [hdrf_out]\n";
        return 1;
    }
    std::string in_path = argv[1];
    std::string out_path = argv[2];
    int width = argc > 3 ? std::atoi(argv[3]) : 800;
    int height = argc > 4 ? std::atoi(argv[4]) : 600;
    int spp = argc > 5 ? std::atoi(argv[5]) : 64;
    std::string hdrf_path = argc > 6 ? argv[6] : "";
    width = std::clamp(width, 64, 1920);
    height = std::clamp(height, 64, 1080);
    spp = std::clamp(spp, 1, 512);

    Scene scene;
    std::cerr << "[C++] Loading " << in_path << "...\n";
    bool ok = false;
    if (in_path.size() >= 7 && in_path.substr(in_path.size() - 7) == ".ntmesh")
        ok = load_ntmesh(in_path, scene);
    else
        ok = load_obj(in_path, scene);
    if (!ok) {
        std::cerr << "Failed to load mesh\n";
        return 2;
    }
    std::cerr << "[C++] Tris: " << scene.tris.size() << " — BVH...\n";
    scene.build();

    Vec3 bmin = scene.subject.bmin, bmax = scene.subject.bmax;
    Vec3 center = (bmin + bmax) * 0.5;
    // Frame full subject (head-to-toe), slight downward look
    center.y = bmin.y + (bmax.y - bmin.y) * 0.48;
    double extent_y = std::max(0.15, bmax.y - bmin.y);
    double extent_xz = std::max(bmax.x - bmin.x, bmax.z - bmin.z);
    double radius = std::max(extent_y * 1.05, extent_xz * 0.85);
    Vec3 eye = center + Vec3(radius * 1.05, radius * 0.55, radius * 1.65);
    Vec3 forward = (center - eye).normalized();
    Vec3 right = forward.cross(Vec3(0, 1, 0)).normalized();
    Vec3 up = right.cross(forward).normalized();
    double fov = 35.0 * M_PI / 180.0;
    double aspect = double(width) / height;
    double scale = std::tan(fov * 0.5);

    AreaLight light;
    light.center = center + Vec3(-radius * 0.2, radius * 2.2, radius * 0.3);
    light.u = Vec3(radius * 0.55, 0, 0);
    light.v = Vec3(0, 0, radius * 0.4);
    light.area = (2 * light.u.length()) * (2 * light.v.length());
    light.emit = Vec3(28.0, 26.0, 22.0);

    std::vector<Vec3> img(width * height);
    unsigned nthreads = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<int> next_row{0};
    std::mutex log_mu;
    int depth = spp >= 64 ? 8 : 6;
    std::cerr << "[C++] Ultra photoreal " << width << "x" << height << " @ " << spp
              << " spp, depth=" << depth << ", threads=" << nthreads << "\n";

    auto worker = [&]() {
        for (;;) {
            int y = next_row.fetch_add(1);
            if (y >= height) break;
            int strata = std::max(1, (int)std::sqrt((double)spp));
            for (int x = 0; x < width; ++x) {
                Vec3 sum(0, 0, 0);
                RNG rng(((uint64_t)y * 1973ull + (uint64_t)x * 9277ull + 1ull) * 0x9E3779B97F4A7C15ull);
                for (int s = 0; s < spp; ++s) {
                    int sx = s % strata, sy = (s / strata) % strata;
                    double jx = (sx + rng.uniform()) / strata;
                    double jy = (sy + rng.uniform()) / strata;
                    double u = (2.0 * ((x + jx) / width) - 1.0) * aspect * scale;
                    double v = (1.0 - 2.0 * ((y + jy) / height)) * scale;
                    Vec3 dir = (forward + right * u + up * v).normalized();
                    sum = sum + trace(scene, light, Ray(eye, dir), rng, depth);
                }
                img[(height - 1 - y) * width + x] = sum * (1.0 / spp);
            }
            if (y % 16 == 0) {
                std::lock_guard<std::mutex> lk(log_mu);
                std::cerr << "  [C++] row " << y << "/" << height << "\n";
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned i = 0; i < nthreads; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();

    if (!write_bmp(out_path, width, height, img)) {
        std::cerr << "Failed to write BMP\n";
        return 3;
    }
    std::cerr << "[C++] Wrote " << out_path << "\n";
    if (!hdrf_path.empty()) {
        if (!write_hdrf(hdrf_path, width, height, img)) {
            std::cerr << "Failed to write HDRF\n";
            return 4;
        }
        std::cerr << "[C++] Wrote " << hdrf_path << "\n";
    }
    return 0;
}
