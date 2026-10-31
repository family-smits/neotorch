/* NeoTorch meshpack — C: OBJ → binary .ntmesh for the C++ path tracer */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float x, y, z; } V3;

static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return NULL; }
    char* buf = (char*)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = 0;
    if (out_len) *out_len = got;
    return buf;
}

static int parse_face_idx(const char* tok, int nverts) {
    int vi = atoi(tok);
    if (vi < 0) vi = nverts + vi + 1;
    return vi - 1;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: meshpack <in.obj> <out.ntmesh>\n");
        return 1;
    }
    size_t len = 0;
    char* text = read_file(argv[1], &len);
    if (!text) {
        fprintf(stderr, "Cannot read %s\n", argv[1]);
        return 2;
    }

    int vcap = 1024, fcap = 1024;
    int nv = 0, nf = 0;
    V3* verts = (V3*)malloc(sizeof(V3) * vcap);
    V3* colors = (V3*)malloc(sizeof(V3) * vcap);
    int* faces = (int*)malloc(sizeof(int) * fcap * 3);
    if (!verts || !colors || !faces) { free(text); return 3; }

    char* line = text;
    while (line && *line) {
        char* nl = strchr(line, '\n');
        if (nl) *nl = 0;
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z, r = 0.72f, g = 0.72f, b = 0.74f;
            int n = sscanf(line + 2, "%f %f %f %f %f %f", &x, &y, &z, &r, &g, &b);
            if (n >= 3) {
                if (nv >= vcap) {
                    vcap *= 2;
                    verts = (V3*)realloc(verts, sizeof(V3) * vcap);
                    colors = (V3*)realloc(colors, sizeof(V3) * vcap);
                }
                verts[nv].x = x; verts[nv].y = y; verts[nv].z = z;
                if (r < 0) r = 0; if (r > 1) r = 1;
                if (g < 0) g = 0; if (g > 1) g = 1;
                if (b < 0) b = 0; if (b > 1) b = 1;
                colors[nv].x = r; colors[nv].y = g; colors[nv].z = b;
                nv++;
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            char tok[64];
            int idx[64], ni = 0;
            const char* p = line + 2;
            while (*p && ni < 64) {
                while (*p && isspace((unsigned char)*p)) p++;
                if (!*p) break;
                int k = 0;
                while (*p && !isspace((unsigned char)*p) && k < 63) tok[k++] = *p++;
                tok[k] = 0;
                idx[ni++] = parse_face_idx(tok, nv);
            }
            for (int i = 1; i + 1 < ni; i++) {
                if (nf >= fcap) {
                    fcap *= 2;
                    faces = (int*)realloc(faces, sizeof(int) * fcap * 3);
                }
                faces[nf * 3 + 0] = idx[0];
                faces[nf * 3 + 1] = idx[i];
                faces[nf * 3 + 2] = idx[i + 1];
                nf++;
            }
        }
        if (!nl) break;
        line = nl + 1;
    }
    free(text);

    if (nv < 3 || nf < 1) {
        fprintf(stderr, "Empty mesh\n");
        return 4;
    }

    /* smooth normals */
    V3* norms = (V3*)calloc((size_t)nv, sizeof(V3));
    for (int i = 0; i < nf; i++) {
        int a = faces[i * 3], b = faces[i * 3 + 1], c = faces[i * 3 + 2];
        if (a < 0 || b < 0 || c < 0 || a >= nv || b >= nv || c >= nv) continue;
        V3 e1 = {verts[b].x - verts[a].x, verts[b].y - verts[a].y, verts[b].z - verts[a].z};
        V3 e2 = {verts[c].x - verts[a].x, verts[c].y - verts[a].y, verts[c].z - verts[a].z};
        V3 n = {e1.y * e2.z - e1.z * e2.y, e1.z * e2.x - e1.x * e2.z, e1.x * e2.y - e1.y * e2.x};
        norms[a].x += n.x; norms[a].y += n.y; norms[a].z += n.z;
        norms[b].x += n.x; norms[b].y += n.y; norms[b].z += n.z;
        norms[c].x += n.x; norms[c].y += n.y; norms[c].z += n.z;
    }
    for (int i = 0; i < nv; i++) {
        float L = sqrtf(norms[i].x * norms[i].x + norms[i].y * norms[i].y + norms[i].z * norms[i].z);
        if (L > 1e-12f) { norms[i].x /= L; norms[i].y /= L; norms[i].z /= L; }
        else { norms[i].x = 0; norms[i].y = 1; norms[i].z = 0; }
    }

    FILE* out = fopen(argv[2], "wb");
    if (!out) { fprintf(stderr, "Cannot write %s\n", argv[2]); return 5; }
    fwrite("NTM1", 1, 4, out);
    uint32_t unv = (uint32_t)nv, unf = (uint32_t)nf;
    fwrite(&unv, 4, 1, out);
    fwrite(&unf, 4, 1, out);
    fwrite(verts, sizeof(V3), (size_t)nv, out);
    fwrite(colors, sizeof(V3), (size_t)nv, out);
    fwrite(norms, sizeof(V3), (size_t)nv, out);
    fwrite(faces, sizeof(int), (size_t)nf * 3, out);
    fclose(out);

    fprintf(stderr, "meshpack: %d verts, %d tris -> %s\n", nv, nf, argv[2]);
    free(verts); free(colors); free(norms); free(faces);
    return 0;
}
