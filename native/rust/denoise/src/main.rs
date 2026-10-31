//! NeoTorch Rust denoise — HDRF → bilateral filter + ACES + BMP/PNG-ready BMP
use std::env;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

fn aces(v: f32) -> f32 {
    let a = 2.51f32;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    ((v * (a * v + b)) / (v * (c * v + d) + e)).clamp(0.0, 1.0)
}

fn luminance(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

fn bilateral_denoise(w: usize, h: usize, src: &[f32], spatial: f32, range: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    let radius = spatial.ceil() as i32;
    let inv_2s = 1.0 / (2.0 * spatial * spatial);
    let inv_2r = 1.0 / (2.0 * range * range);
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let i = (y as usize * w + x as usize) * 3;
            let cr = src[i];
            let cg = src[i + 1];
            let cb = src[i + 2];
            let cl = luminance(cr, cg, cb);
            let mut wr = 0.0f32;
            let mut wg = 0.0f32;
            let mut wb = 0.0f32;
            let mut wsum = 0.0f32;
            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let j = (ny as usize * w + nx as usize) * 3;
                    let nr = src[j];
                    let ng = src[j + 1];
                    let nb = src[j + 2];
                    let nl = luminance(nr, ng, nb);
                    let ds = (dx * dx + dy * dy) as f32;
                    let dr = (nl - cl) * (nl - cl);
                    let weight = (-ds * inv_2s - dr * inv_2r).exp();
                    wr += nr * weight;
                    wg += ng * weight;
                    wb += nb * weight;
                    wsum += weight;
                }
            }
            if wsum > 1e-8 {
                out[i] = wr / wsum;
                out[i + 1] = wg / wsum;
                out[i + 2] = wb / wsum;
            } else {
                out[i] = cr;
                out[i + 1] = cg;
                out[i + 2] = cb;
            }
        }
    }
    out
}

fn write_bmp(path: &Path, w: usize, h: usize, rgb8: &[u8]) -> std::io::Result<()> {
    let row = (w * 3 + 3) & !3;
    let data = row * h;
    let mut buf = vec![0u8; 14 + 40 + data];
    buf[0] = b'B';
    buf[1] = b'M';
    let size = buf.len() as u32;
    buf[2..6].copy_from_slice(&size.to_le_bytes());
    buf[10] = 54;
    buf[14] = 40;
    buf[18..22].copy_from_slice(&(w as u32).to_le_bytes());
    buf[22..26].copy_from_slice(&(h as u32).to_le_bytes());
    buf[26] = 1;
    buf[28] = 24;
    for y in 0..h {
        for x in 0..w {
            let s = (y * w + x) * 3;
            let o = 54 + y * row + x * 3;
            buf[o] = rgb8[s + 2];
            buf[o + 1] = rgb8[s + 1];
            buf[o + 2] = rgb8[s];
        }
    }
    let mut f = File::create(path)?;
    f.write_all(&buf)?;
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: denoise <in.hdrf> <out.bmp> [spatial] [range]");
        std::process::exit(1);
    }
    let in_path = Path::new(&args[1]);
    let out_path = Path::new(&args[2]);
    let spatial: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2.2);
    let range: f32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.12);

    let mut file = File::open(in_path).expect("open hdrf");
    let mut mag = [0u8; 4];
    file.read_exact(&mut mag).unwrap();
    assert_eq!(&mag, b"HDRF", "bad HDRF magic");
    let mut wh = [0u8; 8];
    file.read_exact(&mut wh).unwrap();
    let w = u32::from_le_bytes(wh[0..4].try_into().unwrap()) as usize;
    let h = u32::from_le_bytes(wh[4..8].try_into().unwrap()) as usize;
    let mut bytes = vec![0u8; w * h * 3 * 4];
    file.read_exact(&mut bytes).expect("read floats");
    let mut src = vec![0f32; w * h * 3];
    for i in 0..src.len() {
        src[i] = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
    }
    eprintln!("[Rust] Denoising {w}x{h} bilateral(s={spatial}, r={range})...");
    let den = bilateral_denoise(w, h, &src, spatial, range);

    // mild unsharp after denoise for micro-contrast (photo look)
    let mut sharp = den.clone();
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let i = (y * w + x) * 3;
            for c in 0..3 {
                let center = den[i + c];
                let blur = (den[i + c - 3]
                    + den[i + c + 3]
                    + den[i + c - w * 3]
                    + den[i + c + w * 3])
                    * 0.25;
                sharp[i + c] = (center + (center - blur) * 0.35).max(0.0);
            }
        }
    }

    let mut rgb8 = vec![0u8; w * h * 3];
    for i in 0..w * h {
        let mut r = aces(sharp[i * 3] * 0.9);
        let mut g = aces(sharp[i * 3 + 1] * 0.9);
        let mut b = aces(sharp[i * 3 + 2] * 0.9);
        r = r.powf(1.0 / 2.2);
        g = g.powf(1.0 / 2.2);
        b = b.powf(1.0 / 2.2);
        rgb8[i * 3] = (r * 255.0).clamp(0.0, 255.0) as u8;
        rgb8[i * 3 + 1] = (g * 255.0).clamp(0.0, 255.0) as u8;
        rgb8[i * 3 + 2] = (b * 255.0).clamp(0.0, 255.0) as u8;
    }
    write_bmp(out_path, w, h, &rgb8).expect("write bmp");
    eprintln!("[Rust] Wrote {}", out_path.display());
}
