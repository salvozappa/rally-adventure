import { defineConfig } from 'vite';

export default defineConfig({
  // Relative, so the same build works at the dev root, at the GitHub Pages
  // project path (/rally-adventure/), and behind a custom domain. Safe here
  // only because there is no client-side routing and no absolute asset fetch —
  // every texture is generated at runtime and Rapier inlines its wasm.
  base: './',
  server: {
    port: 5183,
    host: '127.0.0.1',
    // HMR off deliberately. Nothing here hot-swaps usefully — a shader or a
    // terrain edit forces a full reload anyway — and while several people are
    // editing src/ at once the reload storm tears down the WebGL context and
    // the AudioContext mid-frame, which reads as phantom "it went silent" or
    // "it went black" bugs. Reload explicitly instead.
    hmr: false,
  },
  build: { target: 'es2022', sourcemap: true },
  optimizeDeps: { exclude: ['@dimforge/rapier3d-compat'] },
});
