# idbvec - Vector Database (WASM + IndexedDB)

A high-performance client-side vector database built with Rust/WebAssembly and IndexedDB for persistence.

## Features

- **🚀 WASM-Accelerated**: Near-native performance for vector operations
- **💾 Persistent**: Automatic IndexedDB persistence
- **🎯 ANN Search**: HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search
- **📊 Distance Metrics**: Cosine similarity, Euclidean distance, dot product
- **🔧 Type-Safe**: Full TypeScript support
- **📦 Zero Runtime Dependencies**: Self-contained WASM module

## Architecture

```
┌─────────────────────────────────┐
│   TypeScript API (wrapper.ts)   │
├─────────────────────────────────┤
│    WASM Module (Rust)           │
│  - HNSW Index                   │
│  - Distance Metrics             │
│  - Vector Operations            │
├─────────────────────────────────┤
│   IndexedDB Storage             │
│  - Persistent State             │
│  - Automatic Serialization      │
└─────────────────────────────────┘
```

## Building

```bash
# Install wasm-pack (if not already installed)
cargo install wasm-pack

# Build WASM modules
./build-wasm.sh

# Outputs:
# - pkg/bundler  (for webpack/rollup/vite)
# - pkg/nodejs   (for Node.js)
# - pkg/web      (for ES modules)
```

## Usage

### Basic Example

```typescript
import { VectorDatabase } from './wrapper'

// Create database
const db = new VectorDatabase({
  name: 'my-vectors',
  dimensions: 384, // e.g., for all-MiniLM-L6-v2 embeddings
  m: 16, // max connections per layer
  efConstruction: 200, // construction quality
})

// Initialize
await db.init()

// Insert vectors
await db.insert(
  'doc1',
  new Float32Array([0.1, 0.2, 0.3, ...]),
  { title: 'Document 1', category: 'tech' }
)

// Search
const results = await db.search(
  queryVector,
  { k: 5, ef: 50 }
)

console.log(results)
// [
//   { id: 'doc1', distance: 0.05, metadata: { title: 'Document 1', ... } },
//   ...
// ]

// Delete
await db.delete('doc1')

// Close
db.close()
```

### Batch Insert

```typescript
const records = [
  { id: 'vec1', vector: new Float32Array([...]), metadata: { ... } },
  { id: 'vec2', vector: new Float32Array([...]), metadata: { ... } },
  // ...
]

await db.insertBatch(records)
```

### Distance Functions

```typescript
import { cosineSimilarity, euclideanDistance, dotProduct } from './wrapper'

const a = new Float32Array([1, 0, 0])
const b = new Float32Array([0, 1, 0])

const similarity = await cosineSimilarity(a, b) // 0.0
const distance = await euclideanDistance(a, b) // 1.414...
const dot = await dotProduct(a, b) // 0.0
```

## Configuration

### VectorDBConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | string | - | Database name (IndexedDB key) |
| `dimensions` | number | - | Vector dimensionality |
| `m` | number | 16 | Max connections per layer (higher = better recall, more memory) |
| `efConstruction` | number | 200 | Construction quality (higher = better index, slower insert) |
| `metric` | string | "euclidean" | Distance metric: "euclidean", "cosine", or "dotproduct" |

### Search Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | number | 10 | Number of nearest neighbors to return |
| `ef` | number | 50 | Search quality (higher = better recall, slower search) |

## HNSW Parameters Guide

### M (Max Connections)
- **8-12**: Low memory, faster search, lower recall
- **16-32**: Balanced (recommended)
- **32+**: High recall, more memory

### ef_construction
- **100**: Fast build, lower quality
- **200**: Balanced (recommended)
- **400+**: Slow build, high quality

### ef (search)
- **k**: Minimum (fast, lower recall)
- **k * 2-5**: Balanced
- **k * 10+**: High recall (slower)

## Performance

Typical performance on modern hardware:

- **Insert**: ~1-10ms per vector (depends on ef_construction)
- **Search**: ~1-5ms for k=10 (depends on ef and database size)
- **Memory**: ~(dimensions * 4 + M * 8) bytes per vector

## Integration with Next.js

1. Copy WASM build to `public/`:

```bash
cp -r rust/idbvec/pkg/bundler public/idbvec-wasm
```

2. Use in a client component:

```tsx
'use client'

import { VectorDatabase } from '@brainwires/idbvec'
import { useEffect, useState } from 'react'

export function VectorSearch() {
  const [db, setDb] = useState<VectorDatabase | null>(null)

  useEffect(() => {
    const initDB = async () => {
      const vectorDB = new VectorDatabase({
        name: 'app-vectors',
        dimensions: 384,
      })
      await vectorDB.init()
      setDb(vectorDB)
    }
    initDB()
  }, [])

  // Use db for search, insert, etc.
}
```

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 15+
- ✅ Edge 90+

Requires:
- WebAssembly support
- IndexedDB support
- ES modules

## License

MIT OR Apache-2.0
