/* NeoTorch JavaScript photoreal viewer */
(function () {
  const photo = document.getElementById("photo");
  const meta = document.getElementById("meta");
  const info = document.getElementById("info");
  const file = document.getElementById("file");
  const params = new URLSearchParams(location.search);
  const src = params.get("img") || "../output/ultra_photo.png";

  function load(url) {
    photo.onload = () => {
      meta.textContent = `${photo.naturalWidth}×${photo.naturalHeight}`;
      info.textContent = url.split(/[\\/]/).pop();
      applyFilmGrain();
    };
    photo.onerror = () => {
      meta.textContent = "geen beeld — kies een bestand";
      info.textContent = url;
    };
    photo.src = url + (url.startsWith("blob:") ? "" : `?t=${Date.now()}`);
  }

  function applyFilmGrain() {
    const c = document.getElementById("fx");
    const w = photo.naturalWidth;
    const h = photo.naturalHeight;
    if (!w || !h) return;
    c.width = w;
    c.height = h;
    const ctx = c.getContext("2d");
    ctx.drawImage(photo, 0, 0);
    const img = ctx.getImageData(0, 0, w, h);
    const d = img.data;
    // subtle vignette + micro contrast (display polish only)
    const cx = w * 0.5, cy = h * 0.42;
    const maxR = Math.hypot(cx, cy);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const r = Math.hypot(x - cx, y - cy) / maxR;
        const v = 1 - Math.pow(r, 2.2) * 0.18;
        d[i] = Math.min(255, d[i] * v);
        d[i + 1] = Math.min(255, d[i + 1] * v);
        d[i + 2] = Math.min(255, d[i + 2] * v);
      }
    }
    ctx.putImageData(img, 0, 0);
    photo.src = c.toDataURL("image/png");
  }

  document.getElementById("fit").onclick = () => {
    photo.style.maxWidth = "100%";
    photo.style.width = "auto";
  };
  document.getElementById("actual").onclick = () => {
    photo.style.maxWidth = "none";
    photo.style.width = photo.naturalWidth + "px";
  };
  file.onchange = () => {
    const f = file.files && file.files[0];
    if (!f) return;
    load(URL.createObjectURL(f));
  };

  load(src);
})();
