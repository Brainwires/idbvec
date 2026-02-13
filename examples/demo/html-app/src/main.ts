import './style.css'
import {
  VectorDatabase,
  cosineSimilarity,
  euclideanDistance,
  dotProduct,
} from '@brainwires/idbvec'
import type { DistanceMetric } from '@brainwires/idbvec'

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

// DOM elements
const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T
const logEl = $<HTMLDivElement>('log')
const statVectors = $<HTMLDivElement>('stat-vectors')
const statMetric = $<HTMLDivElement>('stat-metric')
const statSearchTime = $<HTMLDivElement>('stat-search-time')
const resultsContainer = $<HTMLDivElement>('results-container')
const resultsBody = $<HTMLTableSectionElement>('results-body')
const distanceSection = $<HTMLDivElement>('distance-section')
const distanceOutput = $<HTMLPreElement>('distance-output')
const selectMetric = $<HTMLSelectElement>('select-metric')
const selectQuery = $<HTMLSelectElement>('select-query')

let db: VectorDatabase | null = null

// Logging
function showLog(msg: string, level: 'info' | 'success' | 'error' = 'info') {
  logEl.textContent = msg
  logEl.className = `log ${level}`
  logEl.classList.remove('hidden')
}

function appendLog(msg: string) {
  logEl.textContent = (logEl.textContent || '') + '\n' + msg
}

function updateStats() {
  const count = db ? db.size() : 0
  statVectors.textContent = String(count)
}

// Initialize database
async function initDB(metric: DistanceMetric = 'cosine') {
  try {
    if (db) {
      await db.destroy()
      db = null
    }

    showLog(`Initializing VectorDatabase (metric: ${metric})...`)

    db = new VectorDatabase({
      name: 'idbvec-html-demo',
      dimensions: 4,
      m: 16,
      efConstruction: 200,
      metric,
    })

    await db.init()

    statMetric.textContent = metric
    updateStats()
    showLog(`Database ready. ${db.size()} vectors loaded from IndexedDB.`, 'success')
    enableButtons(true)
  } catch (err) {
    showLog(`Init failed: ${err}`, 'error')
  }
}

// Enable/disable buttons
function enableButtons(enabled: boolean) {
  document.querySelectorAll<HTMLButtonElement>('button').forEach((btn) => {
    btn.disabled = !enabled
  })
}

// Insert sample data
async function insertSamples() {
  if (!db) return

  try {
    showLog('Inserting 10 sample vectors...')

    const start = performance.now()
    await db.insertBatch(
      SAMPLE_DATA.map((item) => ({
        id: item.id,
        vector: new Float32Array(item.vector),
        metadata: item.metadata,
      }))
    )
    const elapsed = (performance.now() - start).toFixed(1)

    updateStats()
    showLog(`Inserted ${SAMPLE_DATA.length} vectors in ${elapsed}ms. Total: ${db.size()}`, 'success')
  } catch (err) {
    showLog(`Insert failed: ${err}`, 'error')
  }
}

// Search
async function runSearch() {
  if (!db) return

  const queryId = selectQuery.value
  const item = SAMPLE_DATA.find((d) => d.id === queryId)
  if (!item) {
    showLog(`Unknown query ID: ${queryId}`, 'error')
    return
  }

  try {
    showLog(`Searching for nearest neighbors of "${queryId}"...`)

    const query = new Float32Array(item.vector)
    const start = performance.now()
    const results = await db.search(query, { k: 5, ef: 50 })
    const elapsed = performance.now() - start

    statSearchTime.textContent = `${elapsed.toFixed(1)}ms`

    // Render results table
    resultsBody.innerHTML = ''
    results.forEach((r, i) => {
      const tr = document.createElement('tr')
      tr.innerHTML = `
        <td>${i + 1}</td>
        <td>${r.id}</td>
        <td>${r.distance.toFixed(6)}</td>
        <td>${r.metadata ? JSON.stringify(r.metadata) : '--'}</td>
      `
      resultsBody.appendChild(tr)
    })
    resultsContainer.classList.remove('hidden')

    showLog(`Found ${results.length} results in ${elapsed.toFixed(2)}ms`, 'success')
  } catch (err) {
    showLog(`Search failed: ${err}`, 'error')
  }
}

// Get vector by ID
async function getVector() {
  if (!db) return

  const queryId = selectQuery.value
  try {
    const result = await db.get(queryId)
    if (result) {
      const vecStr = Array.from(result.vector).map((v) => v.toFixed(2)).join(', ')
      showLog(
        `get("${queryId}"):\n  vector: [${vecStr}]\n  metadata: ${JSON.stringify(result.metadata)}`,
        'success'
      )
    } else {
      showLog(`get("${queryId}"): null (not found)`, 'info')
    }
  } catch (err) {
    showLog(`Get failed: ${err}`, 'error')
  }
}

// Delete vector
async function deleteVector() {
  if (!db) return

  const queryId = selectQuery.value
  try {
    const deleted = await db.delete(queryId)
    updateStats()
    showLog(
      `delete("${queryId}"): ${deleted ? 'removed' : 'not found'}`,
      deleted ? 'success' : 'info'
    )
  } catch (err) {
    showLog(`Delete failed: ${err}`, 'error')
  }
}

// List IDs
function listIds() {
  if (!db) return
  const ids = db.listIds()
  showLog(`Stored IDs (${ids.length}): ${ids.join(', ')}`, 'info')
}

// Test standalone distance functions
async function testDistanceFns() {
  const a = new Float32Array([1, 0, 0, 0])
  const b = new Float32Array([0, 1, 0, 0])

  try {
    const [cos, euc, dot] = await Promise.all([
      cosineSimilarity(a, b),
      euclideanDistance(a, b),
      dotProduct(a, b),
    ])
    const result = `a=[1,0,0,0]  b=[0,1,0,0]\n  cosine_similarity: ${cos.toFixed(6)}\n  euclidean_distance: ${euc.toFixed(6)}\n  dot_product:        ${dot.toFixed(6)}`
    distanceOutput.textContent = result
    distanceSection.classList.remove('hidden')
    showLog('Standalone distance functions OK', 'success')
  } catch (err) {
    showLog(`Distance functions failed: ${err}`, 'error')
  }
}

// Clear database
async function clearDB() {
  if (!db) return

  try {
    await db.clear()
    updateStats()
    resultsContainer.classList.add('hidden')
    showLog('Database cleared.', 'success')
  } catch (err) {
    showLog(`Clear failed: ${err}`, 'error')
  }
}

// Switch metric
async function switchMetric() {
  const metric = selectMetric.value as DistanceMetric
  resultsContainer.classList.add('hidden')
  await initDB(metric)
}

// Wire up event listeners
$('btn-insert').addEventListener('click', insertSamples)
$('btn-list').addEventListener('click', listIds)
$('btn-distance').addEventListener('click', testDistanceFns)
$('btn-clear').addEventListener('click', clearDB)
$('btn-search').addEventListener('click', runSearch)
$('btn-get').addEventListener('click', getVector)
$('btn-delete').addEventListener('click', deleteVector)
selectMetric.addEventListener('change', switchMetric)

// Disable buttons until ready
enableButtons(false)

// Initialize
appendLog('Loading WASM module...')
initDB()
