import { useCallback, useEffect, useRef, useState } from 'react'
import {
  VectorDatabase,
  cosineSimilarity,
  euclideanDistance,
  dotProduct,
} from '@brainwires/idbvec'
import type { DistanceMetric, SearchResult } from '@brainwires/idbvec'
import './App.css'

// Sample data: words with simple semantic vectors
const SAMPLE_DATA: { id: string; vector: number[]; metadata: Record<string, string> }[] = [
  { id: 'cat', vector: [0.9, 0.1, 0.0, 0.8], metadata: { type: 'animal', legs: '4' } },
  { id: 'dog', vector: [0.85, 0.15, 0.0, 0.75], metadata: { type: 'animal', legs: '4' } },
  { id: 'fish', vector: [0.7, 0.0, 0.9, 0.3], metadata: { type: 'animal', legs: '0' } },
  { id: 'bird', vector: [0.8, 0.3, 0.1, 0.5], metadata: { type: 'animal', legs: '2' } },
  { id: 'car', vector: [0.0, 0.9, 0.0, 0.1], metadata: { type: 'vehicle', wheels: '4' } },
  { id: 'truck', vector: [0.05, 0.85, 0.0, 0.15], metadata: { type: 'vehicle', wheels: '6' } },
  { id: 'boat', vector: [0.1, 0.6, 0.8, 0.0], metadata: { type: 'vehicle', wheels: '0' } },
  { id: 'plane', vector: [0.1, 0.7, 0.3, 0.2], metadata: { type: 'vehicle', wheels: '3' } },
  { id: 'shark', vector: [0.6, 0.0, 0.95, 0.2], metadata: { type: 'animal', legs: '0' } },
  { id: 'whale', vector: [0.5, 0.0, 0.9, 0.4], metadata: { type: 'animal', legs: '0' } },
]

type LogLevel = 'info' | 'success' | 'error'

function App() {
  const dbRef = useRef<VectorDatabase | null>(null)
  const [ready, setReady] = useState(false)
  const [metric, setMetric] = useState<DistanceMetric>('cosine')
  const [vectorCount, setVectorCount] = useState(0)
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [log, setLog] = useState('')
  const [logLevel, setLogLevel] = useState<LogLevel>('info')
  const [searchQuery, setSearchQuery] = useState('cat')
  const [lastSearchMs, setLastSearchMs] = useState<number | null>(null)
  const [distanceResult, setDistanceResult] = useState<string | null>(null)

  const appendLog = useCallback((msg: string, level: LogLevel = 'info') => {
    setLog((prev) => (prev ? prev + '\n' : '') + msg)
    setLogLevel(level)
  }, [])

  // Initialize database
  const initDB = useCallback(async () => {
    try {
      // Destroy previous instance if exists
      if (dbRef.current) {
        await dbRef.current.destroy()
        dbRef.current = null
      }

      setLog('')
      appendLog(`Initializing VectorDatabase (metric: ${metric})...`)

      const db = new VectorDatabase({
        name: 'idbvec-react-demo',
        dimensions: 4,
        m: 16,
        efConstruction: 200,
        metric,
      })

      await db.init()
      dbRef.current = db

      setReady(true)
      setVectorCount(db.size())
      setSearchResults([])
      appendLog(`Database ready. ${db.size()} vectors loaded from IndexedDB.`, 'success')
    } catch (err) {
      appendLog(`Init failed: ${err}`, 'error')
    }
  }, [metric, appendLog])

  useEffect(() => {
    initDB()
    return () => {
      dbRef.current?.close()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Insert sample data
  const insertSamples = useCallback(async () => {
    const db = dbRef.current
    if (!db) return

    try {
      setLog('')
      appendLog('Inserting 10 sample vectors...')

      const start = performance.now()
      await db.insertBatch(
        SAMPLE_DATA.map((item) => ({
          id: item.id,
          vector: new Float32Array(item.vector),
          metadata: item.metadata,
        }))
      )
      const elapsed = (performance.now() - start).toFixed(1)

      const count = db.size()
      setVectorCount(count)
      appendLog(`Inserted ${SAMPLE_DATA.length} vectors in ${elapsed}ms. Total: ${count}`, 'success')
    } catch (err) {
      appendLog(`Insert failed: ${err}`, 'error')
    }
  }, [appendLog])

  // Search
  const runSearch = useCallback(async () => {
    const db = dbRef.current
    if (!db) return

    const item = SAMPLE_DATA.find((d) => d.id === searchQuery)
    if (!item) {
      appendLog(`Unknown query ID: ${searchQuery}`, 'error')
      return
    }

    try {
      setLog('')
      appendLog(`Searching for nearest neighbors of "${searchQuery}"...`)

      const query = new Float32Array(item.vector)
      const start = performance.now()
      const results = await db.search(query, { k: 5, ef: 50 })
      const elapsed = performance.now() - start

      setSearchResults(results)
      setLastSearchMs(elapsed)
      appendLog(
        `Found ${results.length} results in ${elapsed.toFixed(2)}ms`,
        'success'
      )
    } catch (err) {
      appendLog(`Search failed: ${err}`, 'error')
    }
  }, [searchQuery, appendLog])

  // Get vector by ID
  const getVector = useCallback(async () => {
    const db = dbRef.current
    if (!db) return

    try {
      setLog('')
      const result = await db.get(searchQuery)
      if (result) {
        appendLog(
          `get("${searchQuery}"):\n  vector: [${Array.from(result.vector).map((v) => v.toFixed(2)).join(', ')}]\n  metadata: ${JSON.stringify(result.metadata)}`,
          'success'
        )
      } else {
        appendLog(`get("${searchQuery}"): null (not found)`, 'info')
      }
    } catch (err) {
      appendLog(`Get failed: ${err}`, 'error')
    }
  }, [searchQuery, appendLog])

  // Delete vector
  const deleteVector = useCallback(async () => {
    const db = dbRef.current
    if (!db) return

    try {
      setLog('')
      const deleted = await db.delete(searchQuery)
      setVectorCount(db.size())
      appendLog(
        `delete("${searchQuery}"): ${deleted ? 'removed' : 'not found'}`,
        deleted ? 'success' : 'info'
      )
    } catch (err) {
      appendLog(`Delete failed: ${err}`, 'error')
    }
  }, [searchQuery, appendLog])

  // List IDs
  const listIds = useCallback(() => {
    const db = dbRef.current
    if (!db) return

    setLog('')
    const ids = db.listIds()
    appendLog(`Stored IDs (${ids.length}): ${ids.join(', ')}`, 'info')
  }, [appendLog])

  // Test standalone distance functions
  const testDistanceFns = useCallback(async () => {
    const a = new Float32Array([1, 0, 0, 0])
    const b = new Float32Array([0, 1, 0, 0])

    try {
      setLog('')
      const [cos, euc, dot] = await Promise.all([
        cosineSimilarity(a, b),
        euclideanDistance(a, b),
        dotProduct(a, b),
      ])
      const result = `a=[1,0,0,0]  b=[0,1,0,0]\n  cosine_similarity: ${cos.toFixed(6)}\n  euclidean_distance: ${euc.toFixed(6)}\n  dot_product:        ${dot.toFixed(6)}`
      setDistanceResult(result)
      appendLog('Standalone distance functions OK', 'success')
    } catch (err) {
      appendLog(`Distance functions failed: ${err}`, 'error')
    }
  }, [appendLog])

  // Clear database
  const clearDB = useCallback(async () => {
    const db = dbRef.current
    if (!db) return

    try {
      setLog('')
      await db.clear()
      setVectorCount(0)
      setSearchResults([])
      appendLog('Database cleared.', 'success')
    } catch (err) {
      appendLog(`Clear failed: ${err}`, 'error')
    }
  }, [appendLog])

  // Reinitialize with new metric
  const switchMetric = useCallback(
    async (newMetric: DistanceMetric) => {
      setMetric(newMetric)
      if (dbRef.current) {
        await dbRef.current.destroy()
        dbRef.current = null
      }

      setLog('')
      appendLog(`Switching to ${newMetric} metric...`)

      const db = new VectorDatabase({
        name: 'idbvec-react-demo',
        dimensions: 4,
        m: 16,
        efConstruction: 200,
        metric: newMetric,
      })

      await db.init()
      dbRef.current = db
      setVectorCount(db.size())
      setSearchResults([])
      appendLog(`Database reinitialized with ${newMetric} metric.`, 'success')
    },
    [appendLog]
  )

  return (
    <>
      <h1>
        <span>idbvec</span> React Demo
      </h1>
      <p className="subtitle">WASM-powered vector database with IndexedDB persistence</p>

      {/* Stats */}
      <div className="stats">
        <div className="stat-card">
          <div className="value">{vectorCount}</div>
          <div className="label">Vectors</div>
        </div>
        <div className="stat-card">
          <div className="value">4</div>
          <div className="label">Dimensions</div>
        </div>
        <div className="stat-card">
          <div className="value">{metric}</div>
          <div className="label">Metric</div>
        </div>
        <div className="stat-card">
          <div className="value">{lastSearchMs !== null ? `${lastSearchMs.toFixed(1)}ms` : '--'}</div>
          <div className="label">Search Time</div>
        </div>
      </div>

      {/* Log */}
      {log && <div className={`status ${logLevel}`}>{log}</div>}

      {/* Controls */}
      <div className="section">
        <h2>Database Operations</h2>
        <div className="controls">
          <button onClick={insertSamples} disabled={!ready}>
            Insert Samples
          </button>
          <button className="secondary" onClick={listIds} disabled={!ready}>
            List IDs
          </button>
          <button className="secondary" onClick={testDistanceFns} disabled={!ready}>
            Test Distance Fns
          </button>
          <button className="danger" onClick={clearDB} disabled={!ready}>
            Clear All
          </button>
          <select
            className="metric-select"
            value={metric}
            onChange={(e) => switchMetric(e.target.value as DistanceMetric)}
          >
            <option value="cosine">Cosine</option>
            <option value="euclidean">Euclidean</option>
            <option value="dotproduct">Dot Product</option>
          </select>
        </div>
      </div>

      {/* Search */}
      <div className="section">
        <h2>Vector Search</h2>
        <div className="controls">
          <select
            className="metric-select"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          >
            {SAMPLE_DATA.map((d) => (
              <option key={d.id} value={d.id}>
                {d.id}
              </option>
            ))}
          </select>
          <button onClick={runSearch} disabled={!ready || vectorCount === 0}>
            Search (k=5)
          </button>
          <button className="secondary" onClick={getVector} disabled={!ready}>
            Get by ID
          </button>
          <button className="danger" onClick={deleteVector} disabled={!ready}>
            Delete
          </button>
        </div>

        {searchResults.length > 0 && (
          <table className="results-table">
            <thead>
              <tr>
                <th>#</th>
                <th>ID</th>
                <th>Distance</th>
                <th>Metadata</th>
              </tr>
            </thead>
            <tbody>
              {searchResults.map((r, i) => (
                <tr key={r.id}>
                  <td>{i + 1}</td>
                  <td>{r.id}</td>
                  <td>{r.distance.toFixed(6)}</td>
                  <td>{r.metadata ? JSON.stringify(r.metadata) : '--'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Distance Functions */}
      {distanceResult && (
        <div className="section">
          <h2>Standalone Distance Functions</h2>
          <pre className="status info">{distanceResult}</pre>
        </div>
      )}
    </>
  )
}

export default App
