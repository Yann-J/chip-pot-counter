# Chip pot counter

A small browser app that uses your camera, runs a **YOLOv8 ONNX** model in-browser via [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web), draws boxes on a live overlay, counts detections per class, and sums a **pot value** from per-class chip amounts you configure.

## Requirements

- **Node.js** 20 or newer (matches the GitHub Actions workflow)
- A browser-compatible ONNX detection model and class map file (for example `models/model.onnx` + `models/classes.json`)
- **Camera** access in the browser; for real devices, serve the app over **HTTPS** or use `localhost` so `getUserMedia` is allowed

## Install and run locally

```bash
npm ci
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Grant camera permission when prompted.

### Other commands

| Command | Purpose |
| --- | --- |
| `npm run build` | Production build into `dist/` |
| `npm run preview` | Serve the built `dist/` locally for a smoke test |

The Vite config uses `base: "./"` so the built assets work on **GitHub Pages** project URLs as well as on the dev server.

## Configure the app

All settings are edited in the UI (**gear icon** in the header) and saved in this browser only (**`localStorage`**, key `chip-pot-counter-config-v1`). Inference runs locally in the browser with ONNX Runtime Web.

| Setting | What it does |
| --- | --- |
| **Model URL (.onnx)** | URL/path to the ONNX model file. Relative values like `models/model.onnx` resolve from the current page URL (works with GitHub Pages). |
| **Classes URL** | URL/path to the class mapping JSON file (expected object like `{ "0": "class_a", "1": "class_b" }`). |
| **Model input size** | Input image size expected by the model (default `640`). |
| **Min confidence** | Detections below this score are ignored for overlay, counts, and pot total (`0`–`1`). |
| **Max inferences / sec** | Caps how often frames are sent to the model (`1`–`30`; default in code is `8`). |
| **Chip value by class** | For each **exact** class name your model outputs, set a numeric value per chip. Count × value is summed into **Pot value**. Classes you have not priced count as `0`. Rows for new class names can appear after those classes are seen in the video; you can also use **Add class** to add rows manually. |

After changing model URL, classes URL, or input size, the app reloads the ONNX session. If something is wrong, check the status line under the title for load or inference errors.

## ONNX model tips

- Class names in **Chip value by class** must match your `classes.json` entries exactly (including spelling and case).
- For YOLOv8-seg ONNX exports, masks are decoded from the prototype output and displayed as per-instance overlays.

## Deploy to GitHub Pages

On push to `main` or `master`, [.github/workflows/deploy-pages.yml](.github/workflows/deploy-pages.yml) runs `npm ci`, `npm run build`, and deploys the `dist/` artifact with **GitHub Pages**. Enable Pages in the repository settings if you have not already (source: **GitHub Actions**).
