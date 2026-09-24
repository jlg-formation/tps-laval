import { copyFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

// Le runtime WASM d'onnxruntime-web est servi tel quel depuis public/ort.
const tp4 = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(tp4, "..", "node_modules", "onnxruntime-web", "dist");
const dest = join(tp4, "public", "ort");

mkdirSync(dest, { recursive: true });
for (const file of ["ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"]) {
  copyFileSync(join(src, file), join(dest, file));
}
console.log(`onnxruntime-web copié dans ${dest}`);
