import React, { useState, useEffect, useRef } from 'react';

export interface Person {
  id: number;
  display_name: string | null;
  face_count: number;
  cover_photo_id: number | null;
  cover_face_id?: number | null;
  cover_bbox: string | null;
}

interface PeopleGridProps {
  apiBase?: string;
  onSelectPerson?: (personId: number) => void;
  onUpdateStats?: () => void;
}

export default function PeopleGrid({
  apiBase = 'http://127.0.0.1:8000',
  onSelectPerson,
  onUpdateStats,
}: PeopleGridProps) {
  const [people, setPeople] = useState<Person[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Search & Filter
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [viewFilter, setViewFilter] = useState<'all' | 'named'>('all');

  // Inline Rename
  const [editingId, setEditingId] = useState<number | null>(null);
  const [editName, setEditName] = useState<string>('');
  const [isSaving, setIsSaving] = useState<boolean>(false);

  // Recluster
  const [isReclustering, setIsReclustering] = useState<boolean>(false);

  const inputRef = useRef<HTMLInputElement>(null);

  const fetchPeople = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch(`${apiBase}/api/persons`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data: Person[] = await res.json();
      setPeople(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load people');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPeople();
    // Periodically sync people if background indexing is active
    const interval = setInterval(async () => {
      try {
        const sRes = await fetch(`${apiBase}/api/scan/status`);
        if (sRes.ok) {
          const task = await sRes.json();
          if (task.status === 'running' || task.status === 'completed') {
            const pRes = await fetch(`${apiBase}/api/persons`);
            if (pRes.ok) {
              const data = await pRes.json();
              setPeople(data);
              if (onUpdateStats) onUpdateStats();
            }
          }
        }
      } catch {}
    }, 2000);

    return () => clearInterval(interval);
  }, [apiBase]);

  useEffect(() => {
    if (editingId !== null && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [editingId]);

  const handleStartRename = (person: Person, e: React.MouseEvent) => {
    e.stopPropagation();
    setEditingId(person.id);
    setEditName(person.display_name || '');
  };

  const handleCancelRename = () => {
    setEditingId(null);
    setEditName('');
  };

  const handleSaveRename = async (personId: number) => {
    const trimmed = editName.trim();
    const finalName = trimmed.length > 0 ? trimmed : null;

    try {
      setIsSaving(true);
      const res = await fetch(`${apiBase}/api/persons/${personId}/rename`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: finalName }),
      });
      if (!res.ok) throw new Error(`Failed to rename (HTTP ${res.status})`);
      const updated = await res.json();

      setPeople((prev) =>
        prev.map((p) => (p.id === personId ? { ...p, display_name: updated.display_name } : p))
      );
      setEditingId(null);
      if (onUpdateStats) onUpdateStats();
    } catch (err: any) {
      alert(`Error renaming: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent, personId: number) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSaveRename(personId);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      handleCancelRename();
    }
  };

  const handleRecluster = async () => {
    try {
      setIsReclustering(true);
      const res = await fetch(`${apiBase}/api/recluster`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ min_cluster_size: 2 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await fetchPeople();
      if (onUpdateStats) onUpdateStats();
    } catch (err: any) {
      alert(`Clustering error: ${err.message}`);
    } finally {
      setIsReclustering(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: 'var(--text-secondary)' }}>
        Loading people...
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: '20px', color: '#FF3B30', textAlign: 'center' }}>
        <p>Error: {error}</p>
        <button
          onClick={fetchPeople}
          style={{
            marginTop: '8px',
            padding: '4px 12px',
            backgroundColor: 'var(--accent)',
            color: 'var(--text-on-accent)',
            border: 'none',
            borderRadius: '4px',
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (people.length === 0) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', color: 'var(--text-secondary)', gap: '6px' }}>
        <span style={{ fontSize: '13px' }}>No people detected</span>
        <span style={{ fontSize: '11px' }}>
          Scan a folder containing photos to automatically detect and cluster faces.
        </span>
      </div>
    );
  }

  // Filter & Sort
  const namedCount = people.filter((p) => p.display_name && p.display_name.trim().length > 0).length;

  const filtered = people.filter((p) => {
    if (viewFilter === 'named' && (!p.display_name || p.display_name.trim().length === 0)) {
      return false;
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      const name = (p.display_name || `Unnamed Person #${p.id}`).toLowerCase();
      return name.includes(q);
    }
    return true;
  });

  // Sort named first, then face count descending
  filtered.sort((a, b) => {
    const aNamed = !!(a.display_name && a.display_name.trim());
    const bNamed = !!(b.display_name && b.display_name.trim());
    if (aNamed && !bNamed) return -1;
    if (!aNamed && bNamed) return 1;
    return b.face_count - a.face_count;
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', width: '100%', overflow: 'hidden' }}>
      {/* Native Desktop Toolbar */}
      <div
        style={{
          height: '44px',
          minHeight: '44px',
          maxHeight: '44px',
          borderBottom: '1px solid var(--border-color)',
          backgroundColor: 'var(--bg-app)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 14px',
          gap: '12px',
          boxSizing: 'border-box',
        }}
      >
        {/* Left: Filter & Search */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {/* Segmented control: All / Named */}
          <div
            style={{
              display: 'flex',
              backgroundColor: 'var(--bg-input)',
              borderRadius: '5px',
              padding: '2px',
              height: '28px',
              boxSizing: 'border-box',
            }}
          >
            <button
              onClick={() => setViewFilter('all')}
              style={{
                height: '24px',
                padding: '0 10px',
                border: 'none',
                borderRadius: '4px',
                backgroundColor: viewFilter === 'all' ? 'var(--bg-surface)' : 'transparent',
                color: 'var(--text-primary)',
                fontWeight: viewFilter === 'all' ? 600 : 400,
                fontSize: '11px',
                cursor: 'pointer',
              }}
            >
              All ({people.length})
            </button>
            <button
              onClick={() => setViewFilter('named')}
              style={{
                height: '24px',
                padding: '0 10px',
                border: 'none',
                borderRadius: '4px',
                backgroundColor: viewFilter === 'named' ? 'var(--bg-surface)' : 'transparent',
                color: 'var(--text-primary)',
                fontWeight: viewFilter === 'named' ? 600 : 400,
                fontSize: '11px',
                cursor: 'pointer',
              }}
            >
              Named ({namedCount})
            </button>
          </div>

          {/* Search Input */}
          <input
            type="search"
            placeholder="Filter people..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            style={{
              height: '28px',
              width: '180px',
              padding: '0 10px',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '5px',
              outline: 'none',
              fontSize: '11px',
              boxSizing: 'border-box',
            }}
          />
        </div>

        {/* Right: Re-cluster, Refresh actions & total */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => fetchPeople()}
            title="Refresh people list"
            style={{
              height: '28px',
              padding: '0 10px',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '5px',
              fontSize: '11px',
              fontWeight: 500,
              cursor: 'pointer',
              boxSizing: 'border-box',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
            }}
          >
            <span>🔄</span> Refresh
          </button>
          <button
            disabled={isReclustering}
            onClick={handleRecluster}
            style={{
              height: '28px',
              padding: '0 12px',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '5px',
              fontSize: '11px',
              fontWeight: 500,
              cursor: isReclustering ? 'not-allowed' : 'pointer',
              boxSizing: 'border-box',
            }}
          >
            {isReclustering ? 'Clustering...' : 'Re-cluster'}
          </button>
          <span style={{ fontSize: '11px', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', height: '28px' }}>
            {filtered.length} {filtered.length === 1 ? 'person' : 'people'}
          </span>
        </div>
      </div>

      {/* Main Native People Grid (Apple Photos style) */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', backgroundColor: 'var(--bg-app)' }}>
        {filtered.length === 0 && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: 'var(--text-secondary)', fontSize: '12px' }}>
            No matching people
          </div>
        )}

        {filtered.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
              gap: '16px',
            }}
          >
            {filtered.map((person) => {
              const isEditing = editingId === person.id;
              const isNamed = !!(person.display_name && person.display_name.trim());
              const displayName = isNamed ? person.display_name! : 'Unnamed Person';

              return (
                <div
                  key={person.id}
                  onClick={() => onSelectPerson && onSelectPerson(person.id)}
                  style={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    cursor: onSelectPerson ? 'pointer' : 'default',
                    padding: '8px 4px',
                    borderRadius: '6px',
                    backgroundColor: 'transparent',
                    outline: 'none',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'var(--bg-hover)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  {/* Face Avatar Crop */}
                  <div
                    style={{
                      width: '96px',
                      height: '96px',
                      borderRadius: '50%',
                      overflow: 'hidden',
                      backgroundColor: 'var(--bg-input)',
                      border: isNamed ? '2px solid var(--accent)' : '1px solid var(--border-color)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      position: 'relative',
                    }}
                  >
                    {person.cover_photo_id ? (
                      <img
                        src={person.cover_face_id ? `${apiBase}/api/faces/${person.cover_face_id}/crop` : `${apiBase}/api/photos/${person.cover_photo_id}/file`}
                        alt={displayName}
                        loading="lazy"
                        style={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                          display: 'block',
                        }}
                        onError={(e) => {
                          const img = e.currentTarget;
                          if (person.cover_face_id && !img.dataset.retried) {
                            img.dataset.retried = 'true';
                            img.src = `${apiBase}/api/photos/${person.cover_photo_id}/file`;
                          } else {
                            img.style.display = 'none';
                            const parent = img.parentElement;
                            if (parent && !parent.querySelector('.avatar-fallback')) {
                              const fallback = document.createElement('span');
                              fallback.className = 'avatar-fallback';
                              fallback.style.fontSize = '32px';
                              fallback.style.opacity = '0.4';
                              fallback.textContent = '👤';
                              parent.appendChild(fallback);
                            }
                          }
                        }}
                      />
                    ) : (
                      <span style={{ fontSize: '32px', opacity: 0.4 }}>👤</span>
                    )}
                  </div>

                  {/* Name and Rename Field */}
                  <div
                    style={{
                      marginTop: '8px',
                      width: '100%',
                      textAlign: 'center',
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      gap: '2px',
                    }}
                  >
                    {isEditing ? (
                      <input
                        ref={inputRef}
                        type="text"
                        value={editName}
                        disabled={isSaving}
                        placeholder="Name..."
                        onChange={(e) => setEditName(e.target.value)}
                        onKeyDown={(e) => handleKeyDown(e, person.id)}
                        onBlur={() => handleSaveRename(person.id)}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          width: '100px',
                          height: '22px',
                          padding: '0 4px',
                          fontSize: '11px',
                          textAlign: 'center',
                          backgroundColor: 'var(--bg-input)',
                          color: 'var(--text-primary)',
                          border: '1px solid var(--accent)',
                          borderRadius: '3px',
                          outline: 'none',
                        }}
                      />
                    ) : (
                      <span
                        onClick={(e) => handleStartRename(person, e)}
                        title="Click to rename"
                        style={{
                          fontSize: '12px',
                          fontWeight: isNamed ? 600 : 400,
                          color: isNamed ? 'var(--text-primary)' : 'var(--text-secondary)',
                          whiteSpace: 'nowrap',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          maxWidth: '120px',
                          cursor: 'pointer',
                        }}
                      >
                        {displayName}
                      </span>
                    )}

                    {/* Photo Count */}
                    <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                      {person.face_count} {person.face_count === 1 ? 'photo' : 'photos'}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
