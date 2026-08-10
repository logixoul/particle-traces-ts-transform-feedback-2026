vibe coded.

# Curl-noise particles (Three.js + WebGPU + TypeScript)

One million particles advected by a curl-noise field. All particle state lives in
GPU storage buffers and is stepped by a compute shader; nothing is uploaded or read
back per frame (the WebGPU equivalent of transform feedback).

The shader math (hash, value noise, curl, spawn) is plain WGSL via `wgslFn`; TSL is
used only for buffer plumbing and the sprite material. There is no WebGL2 fallback.

```
npm install
npm run dev
```

Needs a WebGPU-capable browser (Chrome/Edge 113+). Left-drag to orbit, scroll to zoom.

Everything lives in [src/main.ts](src/main.ts).
