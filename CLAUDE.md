After finishing any coding task, give me a short commit message to use for the commit, so I don't have to compose it myself.

DO NOT USE GSTACK's browse skill because it's headless so it's almost useless for this project I think.

## Debugging this demo without asking the user

The browse skill is useless here, but not because it is headless — it is because its
Chromium gets **no WebGPU adapter**, so three.js silently falls back to the WebGL2
backend, which cannot run WGSL at all. A plain `--headless=new` Chromium launched
with no GPU-related flags *does* get a real WebGPU adapter on this machine. That is
enough to run the demo, screenshot it, read GPU buffers back, and dump the generated
shaders — i.e. to debug a black screen without a round trip through the user.

### Launching

Playwright's Chromium is already on disk; do not download another one:

```
~/AppData/Local/ms-playwright/chromium-1208/chrome-win64/chrome.exe
```

Flags: `--headless=new --remote-debugging-port=<port> --user-data-dir=<fresh temp dir>
--no-first-run --no-sandbox`.

**Do not add** `--use-angle=swiftshader`, `--use-vulkan=swiftshader`,
`--enable-unsafe-swiftshader`, or `--use-webgpu-adapter=swiftshader`. Every one of
those *removes* the WebGPU adapter (they are WebGL software-rendering switches).
`--enable-unsafe-webgpu` is not needed either. Fewer flags works; more flags does not.

`navigator.gpu` only exists in a secure context, so probe on `http://localhost:<vite
port>/`, never on `about:blank` — on `about:blank` it is `undefined` and looks like a
capability problem when it is not.

### Driving it

Node 24 has a built-in `WebSocket`, so raw CDP needs **zero dependencies**:

1. Poll `http://127.0.0.1:<port>/json/version` until it answers; take
   `webSocketDebuggerUrl` and open it.
2. `Target.createTarget {url:'about:blank'}` → `Target.attachToTarget {targetId,
   flatten:true}` → use the returned `sessionId` on every later message.
3. `Runtime.enable`, `Page.enable`, `Page.navigate {url}`, then wait a few seconds for
   the animation loop to settle.
4. `Runtime.evaluate {expression, awaitPromise:true, returnByValue:true}` to run
   probes; `Page.captureScreenshot {format:'png'}` returns base64.
5. Collect `Runtime.consoleAPICalled` and `Runtime.exceptionThrown` events for logs.

### The three probes worth knowing

Temporarily hang the objects you need on `window` inside `main()`
(`(window as any).__dbg = { renderer, scene, camera, particles, material, ...buffers }`)
and remove it when done.

- **Read a storage buffer back:** `await renderer.getArrayBufferAsync(buf.value)` →
  `new Float32Array(...)`. Proves whether the compute pass is producing sane data,
  separately from whether it renders.
- **Dump the generated WGSL:** `await renderer.debug.getShaderAsync(scene, camera,
  object)` → `{ vertexShader, fragmentShader }`. This is the highest-value probe by
  far: the black-screen bug was one visible line of generated fragment code.
- **Screenshot**, then read the PNG. A black frame at 60fps with millions of triangles
  submitted means "shaded but invisible", which is a very different bug from "nothing
  drawn".

Bisect appearance bugs with URL query params (`?flat`, `?nodisc`, `?size=0.3`)
switching individual nodes to constants, then strip them out afterwards.

### Two traps that cost real time

- **Stray vite servers.** Vite silently binds the *next free port* if its port is
  taken, so a leftover server from an earlier session keeps serving old code while the
  new one runs elsewhere. Check the port in the server's own output, and confirm the
  running library version in-page (`THREE.REVISION`) rather than trusting
  `node_modules`. Kill leftovers with `Get-NetTCPConnection -LocalPort <p> -State
  Listen` → `Stop-Process`.
- **vec3 storage buffers are padded to 4 floats.** `instancedArray(n,'vec3')` arrives
  through `toAttribute()` as a **vec4** whose `w` is never-written padding (0). Feeding
  that straight into `colorNode` sets alpha to 0 and draws nothing. Take `.xyz`.
