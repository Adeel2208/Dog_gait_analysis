export default function ProgressBar({ pct, status }) {
  const color = status === 'error' ? 'var(--red)'
    : pct >= 100 ? 'var(--green)'
    : 'var(--accent)'

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 13, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
          {status === 'error' ? '✗ Error' :
           pct >= 100 ? '✓ Complete' :
           status === 'queued' ? 'Queued…' :
           <><Spinner /> Running inference…</>}
        </span>
        <span style={{ fontSize: 13, fontWeight: 700, color }}>{pct}%</span>
      </div>
      <div style={{
        height: 8, background: 'var(--surface2)',
        borderRadius: 999, overflow: 'hidden',
        border: '1px solid var(--border)',
      }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: `linear-gradient(90deg, var(--accent), ${color})`,
          borderRadius: 999,
          transition: 'width 0.4s ease',
        }} />
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <span style={{
      display: 'inline-block', width: 12, height: 12,
      border: '2px solid var(--border)',
      borderTopColor: 'var(--accent)',
      borderRadius: '50%',
      animation: 'spin 0.7s linear infinite',
    }} />
  )
}

// inject keyframe once
if (typeof document !== 'undefined' && !document.getElementById('spin-kf')) {
  const s = document.createElement('style')
  s.id = 'spin-kf'
  s.textContent = '@keyframes spin { to { transform: rotate(360deg); } }'
  document.head.appendChild(s)
}
