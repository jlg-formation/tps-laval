import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Angular refuse les assets hors de tp2/ : on copie le runtime WASM dans public/.
const tp2 = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(tp2, "..", "node_modules", "onnxruntime-web", "dist");
const dest = join(tp2, "public", "ort");

mkdirSync(dest, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(join(src, file), join(dest, file));
}
console.log(`onnxruntime-web copié dans ${dest}`);
