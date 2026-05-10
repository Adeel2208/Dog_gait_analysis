import { useRef, useState } from 'react'
import { Upload, Film, X } from 'lucide-react'

const ALLOWED = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm']

function fmt(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

export default function UploadZone({ file, onFile, onClear, disabled }) {
  const inputRef = useRef()
  const [drag, setDrag] = useState(false)

  function pick(f) {
    if (!f) return
    if (!ALLOWED.includes(f.type) && !f.name.match(/\.(mp4|avi|mov|mkv|webm)$/i)) {
      alert('Please select a valid video file (MP4, AVI, MOV, MKV, WEBM)')
      return
    }
    onFile(f)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <div
        onClick={() => !disabled && !file && inputRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={e => {
          e.preventDefault(); setDrag(false)
          if (!disabled) pick(e.dataTransfer.files[0])
        }}
        style={{
          border: `2px dashed ${drag ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          padding: '48px 24px',
          textAlign: 'center',
          cursor: disabled || file ? 'default' : 'pointer',
          background: drag ? 'rgba(88,166,255,0.05)' : 'transparent',
          transition: 'all 0.2s',
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={e => pick(e.target.files[0])}
        />
        <Upload size={40} color="var(--muted)" style={{ marginBottom: 12 }} />
        <p style={{ color: 'var(--text)', fontSize: 15, marginBottom: 6 }}>
          Drag &amp; drop your video, or{' '}
          <span style={{ color: 'var(--accent)', fontWeight: 600 }}>browse</span>
        </p>
        <p style={{ color: 'var(--muted)', fontSize: 13 }}>
          MP4 · AVI · MOV · MKV · WEBM
        </p>
      </div>

      {file && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12,
          background: 'var(--surface2)', borderRadius: 'var(--radius-sm)',
          padding: '10px 14px', border: '1px solid var(--border)',
        }}>
          <Film size={20} color="var(--accent)" />
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ fontWeight: 600, fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {file.name}
            </div>
            <div style={{ color: 'var(--muted)', fontSize: 12 }}>{fmt(file.size)}</div>
          </div>
          {!disabled && (
            <button
              onClick={onClear}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red)', padding: 4 }}
            >
              <X size={16} />
            </button>
          )}
        </div>
      )}
    </div>
  )
}
