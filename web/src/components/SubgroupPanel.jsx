/*
 * Video Synth — real-time collaborative visual art synthesizer.
 * Copyright (C) 2026 Kyle Henderson
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

import { useState } from 'react'
import ParamSlider from './ParamSlider.jsx'

// Normalize an animation/subgroup name for fuzzy matching:
// "STRANGE_ATTRACTOR" and "StrangeAttractor" both become "strangeattractor"
function normalize(s) {
  return (s || '').toLowerCase().replace(/[_\s]/g, '')
}

export default function SubgroupPanel({ params, jumpTarget }) {
  // Group by subgroup
  const subgroups = {}
  for (const p of params) {
    const sg = p.subgroup || ''
    if (!subgroups[sg]) subgroups[sg] = []
    subgroups[sg].push(p)
  }
  const sgNames = Object.keys(subgroups)

  const [activeSg, setActiveSg] = useState(sgNames[0] || '')
  const current = activeSg && subgroups[activeSg] ? activeSg : sgNames[0]
  const visibleParams = subgroups[current] || []

  // Find the subgroup that matches the mixer's active animation source
  const jumpSg = jumpTarget
    ? sgNames.find(sg => normalize(sg) === normalize(jumpTarget)) ?? null
    : null

  return (
    <div className="subgroup-panel">
      {sgNames.length > 1 && (
        <div className="subgroup-select-wrap">
          <select
            className="subgroup-select"
            value={current}
            onChange={e => setActiveSg(e.target.value)}
          >
            {sgNames.map(sg => (
              <option key={sg} value={sg}>
                {sg ? sg.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : '(General)'}
              </option>
            ))}
          </select>
          {jumpSg && (
            <button
              className={`jump-btn${jumpSg === current ? ' jump-btn--active' : ''}`}
              title={`Jump to active: ${jumpSg}`}
              onClick={() => setActiveSg(jumpSg)}
            >
              ↗
            </button>
          )}
        </div>
      )}
      <div className="param-list">
        {visibleParams.map(p => (
          <ParamSlider key={p.name} param={p} />
        ))}
      </div>
    </div>
  )
}
