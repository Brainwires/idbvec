# idbvec - Quick Start Guide

## 🚀 What is This?

A **client-side vector database** built with Rust/WebAssembly that runs entirely in the browser. Features:

- ⚡ **WASM-accelerated** vector operations (near-native speed)
- 💾 **IndexedDB persistence** (survives page refreshes)
- 🎯 **HNSW index** for fast approximate nearest neighbor search
- 📊 **Multiple distance metrics** (cosine, euclidean, dot product)
- 🔧 **Full TypeScript support**

## 📂 Project Structure

```
idbvec/
├── src/
│   ├── lib.rs          # Main WASM bindings & public API
│   ├── distance.rs     # Distance metrics (cosine, euclidean, etc.)
│   ├── hnsw.rs         # HNSW index implementation
│   └── vector.rs       # Vector data structures
├── pkg/
│   ├── bundler/        # WASM for webpack/vite/rollup
│   └── web/            # WASM for ES modules
├── Cargo.toml          # Rust dependencies
├── build-wasm.sh       # Build script
├── wrapper.ts          # TypeScript wrapper with IndexedDB
├── example.html        # Browser demo
└── README.md           # Full documentation
```

## 🛠️ Building

```bash
# Build all WASM targets
./build-wasm.sh

# Or build specific target
wasm-pack build --target web --out-dir pkg/web
wasm-pack build --target bundler --out-dir pkg/bundler
```

## 📝 Usage Examples

### 1. Direct WASM (Browser)

Open `example.html` in a browser to see a live demo. Or use directly:

```html
<script type="module">
  import init, { VectorDB } from './pkg/web/vector_db.js'

  await init()

  const db = new VectorDB(
    384,  // dimensions
    16,   // M (max connections)
    200   // ef_construction
  )

  // Insert vectors
  db.insert('doc1', new Float32Array([0.1, 0.2, ...]), { title: 'Document 1' })

  // Search
  const results = db.search(queryVector, 10, 50)
  console.log(results)
</script>
```

### 2. TypeScript Wrapper (with IndexedDB)

```typescript
import { VectorDatabase } from './wrapper'

// Create database with persistence
const db = new VectorDatabase({
  name: 'my-vectors',
  dimensions: 384,
  m: 16,
  efConstruction: 200
})

await db.init() // Loads from IndexedDB if exists

// Insert vectors (auto-saves to IndexedDB)
await db.insert(
  'doc1',
  new Float32Array([...]),
  { title: 'My Document' }
)

// Search
const results = await db.search(queryVector, { k: 10, ef: 50 })

// Results: [{ id: 'doc1', score: 0.95, metadata: {...} }, ...]
```

### 3. Distance Functions

```typescript
import { cosineSimilarity, euclideanDistance } from './wrapper'

const a = new Float32Array([1, 0, 0])
const b = new Float32Array([0, 1, 0])

const sim = await cosineSimilarity(a, b)  // 0.0
const dist = await euclideanDistance(a, b) // 1.414
```

## 🧪 Testing

1. **Browser Test**: Open `example.html`
2. **CLI Test**:
   ```bash
   cd rust/idbvec
   cargo test
   ```

## 🔧 Configuration Guide

### HNSW Parameters

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `dimensions` | Your embedding size | Vector dimensionality (e.g., 384, 768, 1536) |
| `m` | 16-32 | Max connections per layer. Higher = better recall, more memory |
| `efConstruction` | 200-400 | Build quality. Higher = better index, slower insert |
| `ef` (search) | k*2 to k*10 | Search quality. Higher = better recall, slower search |

### Memory Usage

Approximately: `(dimensions * 4 + M * 8)` bytes per vector

Example: 1000 vectors × 384 dimensions × 4 bytes + connections ≈ 1.5-2 MB

## 🎯 Use Cases

1. **Semantic Search** - Search documents by meaning
2. **RAG (Retrieval Augmented Generation)** - Find relevant context for LLMs
3. **Recommendation Systems** - Similar items/products
4. **Clustering** - Group similar data
5. **Deduplication** - Find duplicate/similar content

## 📊 Performance

On modern hardware (M1/M2, Ryzen 5000+):

- **Insert**: 1-10ms per vector
- **Search**: 1-5ms for k=10 (depends on database size & ef)
- **Memory**: ~4-8 bytes per dimension per vector

## 🔗 Integration

### Next.js

```tsx
'use client'

import { VectorDatabase } from '@/idbvec/wrapper'

export function Search() {
  const [db, setDb] = useState<VectorDatabase>()

  useEffect(() => {
    const init = async () => {
      const vdb = new VectorDatabase({
        name: 'app-vectors',
        dimensions: 384
      })
      await vdb.init()
      setDb(vdb)
    }
    init()
  }, [])

  // Use db...
}
```

### React + Vite

1. Copy WASM to public: `cp -r pkg/bundler public/wasm`
2. Import in component: `import init from '/wasm/vector_db.js'`

## 📚 API Reference

### VectorDB (WASM)

```typescript
class VectorDB {
  constructor(dimensions: number, m: number, ef_construction: number)
  insert(id: string, vector: Float32Array, metadata: any): void
  search(query: Float32Array, k: number, ef: number): SearchResult[]
  delete(id: string): boolean
  size(): number
  serialize(): string
  static deserialize(json: string): VectorDB
}
```

### VectorDatabase (TypeScript Wrapper)

```typescript
class VectorDatabase {
  constructor(config: VectorDBConfig)
  async init(): Promise<void>
  async insert(id: string, vector: Float32Array, metadata?: Record<string, string>): Promise<void>
  async insertBatch(records: VectorRecord[]): Promise<void>
  async search(query: Float32Array, options?: SearchOptions): Promise<SearchResult[]>
  async delete(id: string): Promise<boolean>
  size(): number
  async clear(): Promise<void>
  close(): void
}
```

## 🐛 Troubleshooting

### WASM not loading
- Ensure WASM file is served with correct MIME type
- Check browser DevTools for CORS errors
- Use a local server (not `file://`)

### IndexedDB errors
- Check browser storage quota
- Verify IndexedDB is enabled
- Try clearing browser data

### Performance issues
- Reduce `ef` parameter for faster search
- Use smaller `M` value to reduce memory
- Consider chunking large batches

## 📝 License

MIT OR Apache-2.0

## 🤝 Contributing

Built for BrainWires Studio. Feel free to extend and improve!
