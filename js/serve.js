/* Tiny static server for the JS photoreal viewer */
const http = require("http");
const fs = require("fs");
const path = require("path");
const { exec } = require("child_process");

const APP_ROOT = process.env.NEOTORCH_ROOT
  ? path.resolve(process.env.NEOTORCH_ROOT)
  : path.resolve(__dirname, "..");
const ASSETS = process.env.NEOTORCH_ASSETS
  ? path.resolve(process.env.NEOTORCH_ASSETS)
  : APP_ROOT;
const PORT = Number(process.env.PORT || 8765);
const imgArg = process.argv[2] || path.join(APP_ROOT, "output", "ultra_photo.png");

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".png": "image/png",
  ".bmp": "image/bmp",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".obj": "text/plain",
  ".txt": "text/plain; charset=utf-8",
  ".ico": "image/x-icon",
};

function resolveFile(urlPath) {
  if (urlPath === "/" || urlPath === "/js/viewer.html") {
    return path.join(ASSETS, "js", "viewer.html");
  }
  if (urlPath.startsWith("/js/")) {
    return path.join(ASSETS, urlPath.replace(/^\//, ""));
  }
  // absolute windows path passed as query img=C:/...
  if (/^[A-Za-z]:\//.test(urlPath.replace(/^\//, ""))) {
    return urlPath.replace(/^\//, "");
  }
  const underApp = path.normalize(path.join(APP_ROOT, urlPath.replace(/^\//, "")));
  if (underApp.startsWith(APP_ROOT)) return underApp;
  const underAssets = path.normalize(path.join(ASSETS, urlPath.replace(/^\//, "")));
  if (underAssets.startsWith(ASSETS)) return underAssets;
  return null;
}

const server = http.createServer((req, res) => {
  let urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
  const filePath = resolveFile(urlPath);
  if (!filePath) {
    res.writeHead(403);
    res.end("forbidden");
    return;
  }
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("not found: " + urlPath);
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    res.writeHead(200, { "Content-Type": MIME[ext] || "application/octet-stream" });
    res.end(data);
  });
});

server.listen(PORT, () => {
  const abs = path.resolve(imgArg).replace(/\\/g, "/");
  const page = `http://127.0.0.1:${PORT}/js/viewer.html?img=/${encodeURI(abs)}`;
  console.log("[JS] Photoreal viewer:", page);
  const start =
    process.platform === "win32"
      ? `start "" "${page}"`
      : process.platform === "darwin"
        ? `open "${page}"`
        : `xdg-open "${page}"`;
  exec(start);
});
