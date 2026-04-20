import { defineConfig } from "vite";

// Relative base works for GitHub Pages project sites and local preview.
export default defineConfig({
  base: "./",
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
