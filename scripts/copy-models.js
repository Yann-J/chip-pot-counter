import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "models");
const dest = join(root, "public", "models");

if (!existsSync(src)) {
  throw new Error(`Missing model source directory: ${src}`);
}

const files = readdirSync(src, { withFileTypes: true }).filter((entry) => entry.isFile());
if (files.length === 0) {
  throw new Error(`No model files found in ${src}`);
}

mkdirSync(dest, { recursive: true });
for (const file of files) {
  cpSync(join(src, file.name), join(dest, file.name));
  console.log(`Copied models/${file.name} -> public/models/${file.name}`);
}
