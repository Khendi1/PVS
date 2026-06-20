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

export default function TabPanel({ tabs }) {
  const [active, setActive] = useState(0)
  if (!tabs || tabs.length === 0) return null

  return (
    <div className="tab-panel">
      <div className="tab-bar">
        {tabs.map((tab, i) => (
          <button
            key={tab.label}
            className={`tab-btn${active === i ? ' active' : ''}`}
            onClick={() => setActive(i)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div className="tab-content">
        {tabs[active]?.content}
      </div>
    </div>
  )
}
