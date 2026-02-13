# idbvec

A high-performance client-side vector database built with Rust/WebAssembly and IndexedDB for persistence.

## Features

- **WASM-Accelerated** — Near-native performance for vector operations
- **Persistent** — Automatic IndexedDB persistence with debounced saves
- **HNSW Index** — Approximate nearest neighbor search via Hierarchical Navigable Small World graphs
- **Configurable Distance Metrics** — Cosine, Euclidean, and dot product
- **Type-Safe** — Full TypeScript wrapper with complete type definitions
- **Zero Runtime Dependencies** — Self-contained WASM module
- **Dual Package** — Available on both [crates.io](https://crates.io/crates/idbvec) and [npm](https://www.npmjs.com/package/@brainwires/idbvec)

## Installation

### npm (TypeScript/JavaScript)

```bash
npm install @brainwires/idbvec
```

Requires a bundler with WASM support (Vite, webpack, Rollup). For Vite:

```bash
npm install -D vite-plugin-wasm vite-plugin-top-level-await
```

```ts
// vite.config.ts
import wasm from 'vite-plugin-wasm'
import topLevelAwait from 'vite-plugin-top-level-await'

export default defineConfig({
  plugins: [wasm(), topLevelAwait()],
})
```

### Rust (crate)

```toml
[dependencies]
idbvec = "0.2"
```

## Quick Start

```typescript
import { VectorDatabase } from '@brainwires/idbvec'

// Create and initialize
const db = new VectorDatabase({
  name: 'my-vectors',
  dimensions: 384,
  metric: 'cosine',
})
await db.init()

// Insert vectors with metadata
await db.insert(
  'doc1',
  new Float32Array([0.1, 0.2, 0.3 /* ... */]),
  { title: 'Document 1', category: 'tech' }
)

// Search for nearest neighbors
const results = await db.search(
  new Float32Array([0.15, 0.25, 0.35 /* ... */]),
  { k: 5, ef: 50 }
)
// => [{ id: 'doc1', distance: 0.023, metadata: { title: 'Document 1', ... } }, ...]

// Clean up
db.close()
```

## API Reference

### VectorDatabase

The main class providing IndexedDB-backed vector storage.

#### Constructor

```typescript
new VectorDatabase(config: VectorDBConfig)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `string` | *required* | Database name (IndexedDB store key) |
| `dimensions` | `number` | *required* | Vector dimensionality |
| `m` | `number` | `16` | Max connections per HNSW layer |
| `efConstruction` | `number` | `200` | Index build quality |
| `metric` | `DistanceMetric` | `'euclidean'` | `'euclidean'`, `'cosine'`, or `'dotproduct'` |

#### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `init()` | `Promise<void>` | Load WASM module + restore state from IndexedDB |
| `insert(id, vector, metadata?)` | `Promise<void>` | Insert or upsert a vector |
| `insertBatch(records)` | `Promise<void>` | Batch insert multiple vectors |
| `search(query, options?)` | `Promise<SearchResult[]>` | k-NN search (returns `{ id, distance, metadata }`) |
| `get(id)` | `Promise<GetResult \| null>` | Retrieve a vector and its metadata by ID |
| `has(id)` | `boolean` | Check if a vector exists |
| `listIds()` | `string[]` | List all stored vector IDs |
| `delete(id)` | `Promise<boolean>` | Delete a vector by ID |
| `deleteBatch(ids)` | `Promise<number>` | Delete multiple vectors, returns count removed |
| `size()` | `number` | Total number of stored vectors |
| `clear()` | `Promise<void>` | Remove all vectors |
| `flush()` | `Promise<void>` | Force-write pending changes to IndexedDB |
| `exportData()` | `string` | Serialize entire database to JSON |
| `importData(json)` | `Promise<void>` | Restore database from JSON |
| `destroy()` | `Promise<void>` | Delete the IndexedDB database entirely |
| `close()` | `void` | Release WASM memory and close IndexedDB |

#### Search Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `number` | `10` | Number of nearest neighbors to return |
| `ef` | `number` | `50` | Search quality (higher = better recall, slower) |

### Standalone Distance Functions

```typescript
import { cosineSimilarity, euclideanDistance, dotProduct } from '@brainwires/idbvec'

const a = new Float32Array([1, 0, 0])
const b = new Float32Array([0, 1, 0])

await cosineSimilarity(a, b)  // 0.0 (orthogonal)
await euclideanDistance(a, b)  // 1.414... (√2)
await dotProduct(a, b)         // 0.0
```

### Input Validation

- Dimension mismatches throw errors
- `NaN` and `Infinity` values are rejected on insert
- Duplicate IDs upsert (replace the existing vector)

## Architecture

```
┌─────────────────────────────────┐
│   TypeScript API (wrapper.ts)   │  VectorDatabase class
│   - IndexedDB persistence       │  - Debounced auto-save
│   - Async init/search/insert    │  - Export/import
├─────────────────────────────────┤
│    WASM Module (Rust)           │  VectorDB struct
│   - HNSW graph index            │  - Multi-layer search
│   - Distance metrics            │  - Serialize/deserialize
│   - Input validation            │  - NaN/Inf rejection
├─────────────────────────────────┤
│   IndexedDB                     │  Browser storage
│   - Automatic state restore     │  - Survives page reloads
└─────────────────────────────────┘
```

## HNSW Tuning Guide

### M (Max Connections per Layer)

| Value | Trade-off |
|-------|-----------|
| 8–12 | Low memory, faster search, lower recall |
| **16–32** | **Balanced (recommended)** |
| 32+ | High recall, more memory |

### ef_construction (Build Quality)

| Value | Trade-off |
|-------|-----------|
| 100 | Fast build, lower quality |
| **200** | **Balanced (recommended)** |
| 400+ | Slow build, high quality |

### ef (Search Quality)

| Value | Trade-off |
|-------|-----------|
| k | Minimum (fast, lower recall) |
| **k × 2–5** | **Balanced** |
| k × 10+ | High recall, slower |

## Performance

Typical on modern hardware:

| Operation | Time |
|-----------|------|
| Insert | ~1–10ms per vector |
| Search (k=10) | ~1–5ms |
| Memory | ~(dimensions × 4 + M × 8) bytes per vector |

## Examples

Working demos are in [`examples/demo/`](examples/demo/):

| Demo | Stack | Run |
|------|-------|-----|
| [`react-app`](examples/demo/react-app/) | React + TypeScript + Vite | `cd examples/demo/react-app && npm install && npm run dev` |
| [`html-app`](examples/demo/html-app/) | Vanilla TypeScript + Vite | `cd examples/demo/html-app && npm install && npm run dev` |

Both demos exercise: insert, search, get, delete, metric switching, standalone distance functions, and IndexedDB persistence.

### Framework Integration (Next.js)

Use in a client component — WASM cannot run server-side:

```tsx
'use client'

import { VectorDatabase } from '@brainwires/idbvec'
import { useEffect, useRef } from 'react'

export function VectorSearch() {
  const db = useRef<VectorDatabase | null>(null)

  useEffect(() => {
    const vectorDB = new VectorDatabase({
      name: 'app-vectors',
      dimensions: 384,
      metric: 'cosine',
    })
    vectorDB.init().then(() => {
      db.current = vectorDB
    })
    return () => db.current?.close()
  }, [])
}
```

## Building from Source

```bash
# Prerequisites
cargo install wasm-pack

# Build all WASM targets
./build-wasm.sh
# => pkg/bundler/, pkg/nodejs/, pkg/web/

# Build npm package (bundler target + TypeScript wrapper)
./build-npm.sh

# Run native tests (54 unit + 5 integration)
cargo test

# Run WASM tests (26 browser tests, requires Chrome)
wasm-pack test --headless --chrome

# Lint
cargo clippy
```

## Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 15+
- Edge 90+

Requires WebAssembly, IndexedDB, and ES module support.

## License

MIT OR Apache-2.0
