// Change this if running the API on a different host/port.
// In dev mode (npm run dev) the Vite proxy handles routing, so ''.
// If opening dist/index.html directly, set this to 'http://127.0.0.1:8000'.
export const API_BASE = ''

export async function fetchParams() {
  const r = await fetch(`${API_BASE}/params`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function setParam(name, value) {
  const r = await fetch(`${API_BASE}/params/${encodeURIComponent(name)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value: parseFloat(value) }),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function resetParam(name) {
  const r = await fetch(`${API_BASE}/params/reset/${encodeURIComponent(name)}`, {
    method: 'POST',
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function patchAction(action) {
  const r = await fetch(`${API_BASE}/patch/${action}`, { method: 'POST' })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function midiLearnStart(param) {
  const r = await fetch(`${API_BASE}/midi/learn`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ param }),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function midiLearnCancel() {
  const r = await fetch(`${API_BASE}/midi/learn/cancel`, { method: 'POST' })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function midiLearnStatus() {
  const r = await fetch(`${API_BASE}/midi/learn/status`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export function streamUrl() {
  // Cache-bust on each call to force reconnect
  return `${API_BASE}/stream?t=${Date.now()}`
}

export function wsStreamUrl() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const base = API_BASE || `${proto}://${location.host}`
  return `${base}/ws/stream`
}

export async function fetchAudioBands() {
  const r = await fetch(`${API_BASE}/audio/bands`)
  if (!r.ok) throw new Error(r.status)
  return r.json()
}

export async function fetchLfos() {
  const r = await fetch(`${API_BASE}/lfo`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function fetchLfo(paramName) {
  const r = await fetch(`${API_BASE}/lfo/${encodeURIComponent(paramName)}`)
  if (!r.ok) return null
  return r.json()
}

export async function createLfo(paramName, config) {
  const r = await fetch(`${API_BASE}/lfo/${encodeURIComponent(paramName)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function updateLfo(paramName, config) {
  const r = await fetch(`${API_BASE}/lfo/${encodeURIComponent(paramName)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export async function deleteLfo(paramName) {
  const r = await fetch(`${API_BASE}/lfo/${encodeURIComponent(paramName)}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}
