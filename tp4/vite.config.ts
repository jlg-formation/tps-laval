import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const here = dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  root: here,
  // Le runtime WASM est servi depuis public/ort : inutile (et nuisible) de le pré-bundler.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
