# Chip pot counter

A small browser app that uses your camera, runs a **Roboflow** object-detection model via [inferencejs](https://github.com/roboflow/inferencejs), draws boxes on a live overlay, counts detections per class, and sums a **pot value** from per-class chip amounts you configure.

## Requirements

- **Node.js** 20 or newer (matches the GitHub Actions workflow)
- A **Roboflow** project with a deployed model suitable for browser inference (publishable API key + model identifier)
- **Camera** access in the browser; for real devices, serve the app over **HTTPS** or use `localhost` so `getUserMedia` is allowed

## Install and run locally

```bash
npm ci
npm run dev
```

Open the URL Vite prints (usually `http://localhost:5173`). Grant camera permission when prompted.

### Other commands

| Command           | Purpose                                      |
| ----------------- | -------------------------------------------- |
| `npm run build`   | Production build into `dist/`                |
| `npm run preview` | Serve the built `dist/` locally for a smoke test |

The Vite config uses `base: "./"` so the built assets work on **GitHub Pages** project URLs as well as on the dev server.

## Configure the app

All settings are edited in the UI (**gear icon** in the header) and saved in this browser only (**`localStorage`**, key `chip-pot-counter-config-v1`). Nothing is sent to your own server; inference goes to Roboflow’s hosted inference as used by inferencejs.

| Setting                 | What it does |
| ----------------------- | ------------ |
| **Publishable API key** | Roboflow publishable key (starts with `rf_`). |
| **Model slug**          | Identifier in the form `workspace/project` as shown in Roboflow for the model you want to run. |
| **Model version**       | Numeric version of the deployed model (minimum `1`). |
| **Min confidence**      | Detections below this score are ignored for overlay, counts, and pot total (`0`–`1`). |
| **Max inferences / sec** | Caps how often frames are sent to the model (`1`–`30`; default in code is `8`). |
| **Chip value by class** | For each **exact** class name your model outputs, set a numeric value per chip. Count × value is summed into **Pot value**. Classes you have not priced count as `0`. Rows for new class names can appear after those classes are seen in the video; you can also use **Add class** to add rows manually. |

After changing key, model, or version, the app restarts the inference worker. If something is wrong, check the status line under the title for load or inference errors.

## Roboflow model tips

- Class names in **Chip value by class** must match the model’s class strings exactly (including spelling and case).
- The overlay expects **bounding boxes** on each detection (normalized `0`–`1` or pixel coordinates); that matches typical object-detection exports.

## Deploy to GitHub Pages

On push to `main` or `master`, [.github/workflows/deploy-pages.yml](.github/workflows/deploy-pages.yml) runs `npm ci`, `npm run build`, and deploys the `dist/` artifact with **GitHub Pages**. Enable Pages in the repository settings if you have not already (source: **GitHub Actions**).

## Optional: `package.json` override

This repo includes an npm **override** for `@rollup/rollup-linux-arm64-gnu` pointing at a local stub under `stubs/`. That avoids optional native Rollup binary resolution issues on some environments. You normally do not need to change it for local development or CI.
