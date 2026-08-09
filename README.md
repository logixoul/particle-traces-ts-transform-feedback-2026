vibe coded.

# Curl-noise particles (Three.js + WebGPU + TypeScript)

One million particles advected by a curl-noise field. All particle state lives in
GPU storage buffers and is stepped by a compute shader; nothing is uploaded or read
back per frame (the WebGPU equivalent of transform feedback).

```
npm install
npm run dev
```

Needs a WebGPU-capable browser (Chrome/Edge 113+). Left-drag to orbit, scroll to zoom.

Everything lives in [src/main.ts](src/main.ts).
