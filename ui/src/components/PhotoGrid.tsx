import React, { useState, useEffect, useCallback } from 'react';
import { Person } from './PeopleGrid';

export interface PhotoFace {
  face_id: number;
  person_id: number | null;
  name: string;
  is_named: boolean;
  bbox?: string;
}

export interface Photo {
  id: number;
  filepath: string;
  date_taken: string | null;
  phash: string | null;
  processing_state: string;
  score?: number;
  created_at?: string;
  people?: PhotoFace[];
}

interface PhotoGridProps {
  apiBase?: string;
  initialPersonId?: number | null;
  onUpdateStats?: () => void;
}

export default function PhotoGrid({
  apiBase = 'http://127.0.0.1:8000',
  initialPersonId = null,
  onUpdateStats,
}: PhotoGridProps) {
  const [photos, setPhotos] = useState<Photo[]>([]);
  const [people, setPeople] = useState<Person[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [textQuery, setTextQuery] = useState<string>('');
  const [selectedPersonId, setSelectedPersonId] = useState<string>(
    initialPersonId ? String(initialPersonId) : ''
  );
  const [dateFrom, setDateFrom] = useState<string>('');
  const [dateTo, setDateTo] = useState<string>('');

  // Selected Photo / Lightbox Inspector
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [editingFaceId, setEditingFaceId] = useState<number | null>(null);
  const [faceInputName, setFaceInputName] = useState<string>('');
  const [isSavingFace, setIsSavingFace] = useState<boolean>(false);

  // Multi-select Split Mode
  const [isSelectMode, setIsSelectMode] = useState<boolean>(false);
  const [selectedPhotoIds, setSelectedPhotoIds] = useState<Set<number>>(new Set());

  const handleRemoveFaceFromPerson = async (faceId: number, personId: number | null) => {
    if (!personId) return;
    if (!confirm('Remove this photo from this person?')) return;
    try {
      const res = await fetch(`${apiBase}/api/persons/${personId}/remove-face`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face_id: faceId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      fetchPhotos();
      fetchPeople();
      if (onUpdateStats) onUpdateStats();
      setSelectedIndex(null);
    } catch (err: any) {
      alert(`Error removing face: ${err.message}`);
    }
  };

  const toggleSelectPhoto = (id: number) => {
    setSelectedPhotoIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const handleSplitSelected = async () => {
    if (!selectedPersonId || selectedPhotoIds.size === 0) return;
    const newName = prompt('Enter a name for the new person (optional):');
    if (newName === null) return;

    const faceIdsToMove: number[] = [];
    photos.forEach((p) => {
      if (selectedPhotoIds.has(p.id) && p.people) {
        p.people.forEach((f) => {
          if (f.person_id === Number(selectedPersonId)) {
            faceIdsToMove.push(f.face_id);
          }
        });
      }
    });

    if (faceIdsToMove.length === 0) {
      alert('No faces matching this person found in the selected photos.');
      return;
    }

    try {
      const res = await fetch(`${apiBase}/api/persons/${selectedPersonId}/split`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          face_ids: faceIdsToMove,
          new_display_name: newName.trim() || undefined,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setSelectedPhotoIds(new Set());
      setIsSelectMode(false);
      fetchPhotos();
      fetchPeople();
      if (onUpdateStats) onUpdateStats();
    } catch (err: any) {
      alert(`Error splitting person: ${err.message}`);
    }
  };

  // Fetch People list for filter
  const fetchPeople = useCallback(async () => {
    try {
      const res = await fetch(`${apiBase}/api/persons`);
      if (res.ok) {
        const data = await res.json();
        setPeople(data);
      }
    } catch {
      // ignore
    }
  }, [apiBase]);

  useEffect(() => {
    fetchPeople();
  }, [fetchPeople]);

  const fetchPhotos = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      if (textQuery.trim()) params.set('text_query', textQuery.trim());
      if (selectedPersonId) params.set('person_id', selectedPersonId);
      if (dateFrom) params.set('date_from', dateFrom);
      if (dateTo) params.set('date_to', dateTo);
      params.set('limit', '5000');

      const qs = params.toString();
      const url = `${apiBase}/api/photos${qs ? `?${qs}` : ''}`;

      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Photo[] = await res.json();
      setPhotos(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load photos');
    } finally {
      setLoading(false);
    }
  }, [apiBase, textQuery, selectedPersonId, dateFrom, dateTo]);

  useEffect(() => {
    fetchPhotos();
  }, [fetchPhotos]);

  const handleSearchKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      fetchPhotos();
    }
  };

  const handleClearFilters = () => {
    setTextQuery('');
    setSelectedPersonId('');
    setDateFrom('');
    setDateTo('');
  };

  const selectedPhoto = selectedIndex !== null && photos[selectedIndex] ? photos[selectedIndex] : null;

  const handleSaveFaceName = async (faceId: number) => {
    const trimmed = faceInputName.trim();
    if (!trimmed) return;

    try {
      setIsSavingFace(true);
      const res = await fetch(`${apiBase}/api/faces/${faceId}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: trimmed }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setPhotos((prev) =>
        prev.map((p) => {
          if (!p.people) return p;
          return {
            ...p,
            people: p.people.map((f) =>
              f.face_id === faceId
                ? { ...f, name: data.name, is_named: true, person_id: data.person_id }
                : f
            ),
          };
        })
      );

      setEditingFaceId(null);
      setFaceInputName('');
      fetchPeople();
      if (onUpdateStats) onUpdateStats();
    } catch (err: any) {
      alert(`Error saving name: ${err.message}`);
    } finally {
      setIsSavingFace(false);
    }
  };

  const hasFilters = !!(textQuery || selectedPersonId || dateFrom || dateTo);

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
        {/* Left: Filters */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flex: 1, overflow: 'hidden' }}>
          {/* Spotlight-style Search Input */}
          <input
            type="text"
            placeholder="Search photos..."
            value={textQuery}
            onChange={(e) => setTextQuery(e.target.value)}
            onKeyDown={handleSearchKeyDown}
            style={{
              height: '28px',
              width: '190px',
              padding: '0 10px',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '5px',
              outline: 'none',
              fontSize: '12px',
              boxSizing: 'border-box',
            }}
          />

          {/* Filter by Person Select */}
          <select
            data-testid="person-filter-select"
            value={selectedPersonId}
            onChange={(e) => setSelectedPersonId(e.target.value)}
            style={{
              height: '28px',
              padding: '0 8px',
              backgroundColor: 'var(--bg-input)',
              color: 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '5px',
              outline: 'none',
              fontSize: '12px',
              boxSizing: 'border-box',
              cursor: 'pointer',
            }}
          >
            <option value="">Filter by person: All</option>
            {people.map((p) => (
              <option key={p.id} value={String(p.id)}>
                {p.display_name || `Unnamed Person #${p.id}`} ({p.face_count})
              </option>
            ))}
          </select>

          {/* Date Range Inputs */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '11px', color: 'var(--text-secondary)' }}>
            <span style={{ userSelect: 'none', fontWeight: 500 }}>From</span>
            <input
              type="date"
              value={dateFrom}
              onChange={(e) => setDateFrom(e.target.value)}
              style={{
                height: '28px',
                padding: '0 6px',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '5px',
                fontSize: '11px',
                boxSizing: 'border-box',
                outline: 'none',
              }}
            />
            <span style={{ userSelect: 'none', fontWeight: 500 }}>To</span>
            <input
              type="date"
              value={dateTo}
              onChange={(e) => setDateTo(e.target.value)}
              style={{
                height: '28px',
                padding: '0 6px',
                backgroundColor: 'var(--bg-input)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
                borderRadius: '5px',
                fontSize: '11px',
                boxSizing: 'border-box',
                outline: 'none',
              }}
            />
          </div>

          {hasFilters && (
            <button
              onClick={handleClearFilters}
              style={{
                height: '28px',
                padding: '0 8px',
                backgroundColor: 'transparent',
                color: 'var(--accent)',
                border: 'none',
                borderRadius: '4px',
                fontSize: '11px',
                cursor: 'pointer',
                display: 'inline-flex',
                alignItems: 'center',
                fontWeight: 500,
              }}
            >
              Clear
            </button>
          )}
        </div>

        {/* Right: Actions & Stats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexShrink: 0 }}>
          {selectedPersonId && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <button
                onClick={() => {
                  setIsSelectMode(!isSelectMode);
                  setSelectedPhotoIds(new Set());
                }}
                style={{
                  height: '28px',
                  padding: '0 10px',
                  backgroundColor: isSelectMode ? 'var(--accent)' : 'var(--bg-input)',
                  color: isSelectMode ? 'var(--text-on-accent)' : 'var(--text-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '5px',
                  fontSize: '11px',
                  fontWeight: 500,
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  boxSizing: 'border-box',
                }}
              >
                {isSelectMode ? '✕ Cancel Selection' : '✂ Select to Split'}
              </button>

              {isSelectMode && selectedPhotoIds.size > 0 && (
                <button
                  onClick={handleSplitSelected}
                  style={{
                    height: '28px',
                    padding: '0 12px',
                    backgroundColor: '#10b981',
                    color: '#ffffff',
                    border: 'none',
                    borderRadius: '5px',
                    fontSize: '11px',
                    fontWeight: 600,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    boxSizing: 'border-box',
                  }}
                >
                  Split ({selectedPhotoIds.size}) into New Person
                </button>
              )}
            </div>
          )}

          <div style={{ fontSize: '11px', color: 'var(--text-secondary)', whiteSpace: 'nowrap', display: 'flex', alignItems: 'center', height: '28px' }}>
            {photos.length} {photos.length === 1 ? 'photo' : 'photos'}
          </div>
        </div>
      </div>

      {/* Main Grid Area: Dense, Minimal Gutters, Zero SaaS Cards */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '2px', backgroundColor: 'var(--bg-app)' }}>
        {loading && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '200px', color: 'var(--text-secondary)' }}>
            Loading photos...
          </div>
        )}

        {error && !loading && (
          <div style={{ padding: '20px', color: '#FF3B30', textAlign: 'center' }}>
            {error}
          </div>
        )}

        {!loading && !error && photos.length === 0 && (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px', color: 'var(--text-secondary)', gap: '8px' }}>
            <span style={{ fontSize: '13px' }}>No photos found</span>
            <span style={{ fontSize: '11px' }}>
              {hasFilters ? 'Try clearing your active filters.' : 'Scan a folder from the sidebar to populate your library.'}
            </span>
          </div>
        )}

        {!loading && photos.length > 0 && (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))',
              gap: '2px',
            }}
          >
            {photos.map((photo, index) => {
              const filename = photo.filepath.split(/[/\\]/).pop() || photo.filepath;
              const isSelected = selectedIndex === index;

              return (
                <div
                  key={photo.id}
                  onClick={() => {
                    if (isSelectMode) {
                      toggleSelectPhoto(photo.id);
                    } else {
                      setSelectedIndex(index);
                    }
                  }}
                  style={{
                    position: 'relative',
                    aspectRatio: '1 / 1',
                    backgroundColor: 'var(--bg-input)',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    outline: selectedPhotoIds.has(photo.id)
                      ? '3px solid #10b981'
                      : isSelected
                      ? '2px solid var(--accent)'
                      : 'none',
                    outlineOffset: '-2px',
                  }}
                >
                  {/* Selection indicator when in select mode */}
                  {isSelectMode && (
                    <div
                      style={{
                        position: 'absolute',
                        top: '6px',
                        right: '6px',
                        zIndex: 10,
                        backgroundColor: selectedPhotoIds.has(photo.id) ? '#10b981' : 'rgba(0, 0, 0, 0.65)',
                        color: '#FFFFFF',
                        width: '20px',
                        height: '20px',
                        borderRadius: '4px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        fontSize: '12px',
                        fontWeight: 700,
                        border: '1px solid rgba(255, 255, 255, 0.8)',
                      }}
                    >
                      {selectedPhotoIds.has(photo.id) ? '✓' : ''}
                    </div>
                  )}
                  <img
                    src={`${apiBase}/api/photos/${photo.id}/file`}
                    alt={filename}
                    loading="lazy"
                    style={{
                      width: '100%',
                      height: '100%',
                      objectFit: 'cover',
                      display: 'block',
                    }}
                    onError={(e) => {
                      (e.target as HTMLElement).style.display = 'none';
                    }}
                  />

                  {/* Semantic match badge if score present */}
                  {photo.score !== undefined && (
                    <div
                      style={{
                        position: 'absolute',
                        top: '4px',
                        left: '4px',
                        backgroundColor: 'var(--accent)',
                        color: 'var(--text-on-accent)',
                        fontSize: '10px',
                        fontWeight: 600,
                        padding: '1px 5px',
                        borderRadius: '3px',
                      }}
                    >
                      {Math.round(photo.score * 100)}% match
                    </div>
                  )}

                  {/* Minimal Metadata Overlay on Bottom */}
                  <div
                    style={{
                      position: 'absolute',
                      bottom: 0,
                      left: 0,
                      right: 0,
                      backgroundColor: 'rgba(0, 0, 0, 0.65)',
                      color: '#FFFFFF',
                      padding: '3px 6px',
                      fontSize: '10px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      overflow: 'hidden',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }} title={filename}>
                      {filename}
                    </span>
                    {photo.people && photo.people.length > 0 && (
                      <span style={{ fontSize: '9px', opacity: 0.9, marginLeft: '4px' }}>
                        👤 {photo.people.map((p) => p.name).join(', ')}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Native Photo Inspector Modal */}
      {selectedPhoto && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.88)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 9999,
          }}
          onClick={() => setSelectedIndex(null)}
        >
          <div
            style={{
              width: '90vw',
              maxWidth: '960px',
              height: '85vh',
              backgroundColor: 'var(--bg-app)',
              borderRadius: '6px',
              border: '1px solid var(--border-color)',
              overflow: 'hidden',
              display: 'flex',
              flexDirection: 'column',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Inspector Top Bar */}
            <div
              style={{
                height: '36px',
                padding: '0 12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid var(--border-color)',
                backgroundColor: 'var(--bg-sidebar)',
              }}
            >
              <span style={{ fontSize: '12px', fontWeight: 500, color: 'var(--text-primary)' }}>
                {selectedPhoto.filepath.split(/[/\\]/).pop()}
              </span>
              <button
                onClick={() => setSelectedIndex(null)}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'var(--text-secondary)',
                  fontSize: '13px',
                  fontWeight: 600,
                }}
              >
                Done
              </button>
            </div>

            {/* Photo Preview */}
            <div
              style={{
                flex: 1,
                backgroundColor: '#000000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                position: 'relative',
              }}
            >
              <img
                src={`${apiBase}/api/photos/${selectedPhoto.id}/file`}
                alt={selectedPhoto.filepath}
                style={{ maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' }}
              />

              {/* Prev / Next Controls */}
              {selectedIndex !== null && selectedIndex > 0 && (
                <button
                  onClick={() => setSelectedIndex(selectedIndex - 1)}
                  style={{
                    position: 'absolute',
                    left: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    color: '#FFFFFF',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '4px',
                    width: '32px',
                    height: '32px',
                    fontSize: '18px',
                  }}
                >
                  ‹
                </button>
              )}
              {selectedIndex !== null && selectedIndex < photos.length - 1 && (
                <button
                  onClick={() => setSelectedIndex(selectedIndex + 1)}
                  style={{
                    position: 'absolute',
                    right: '12px',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    color: '#FFFFFF',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    borderRadius: '4px',
                    width: '32px',
                    height: '32px',
                    fontSize: '18px',
                  }}
                >
                  ›
                </button>
              )}
            </div>

            {/* Inspector Bottom Metadata */}
            <div
              style={{
                padding: '10px 14px',
                borderTop: '1px solid var(--border-color)',
                backgroundColor: 'var(--bg-app)',
                display: 'flex',
                flexDirection: 'column',
                gap: '8px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}>
                <span>Date: {selectedPhoto.date_taken || 'Unknown'}</span>
                <span>Path: {selectedPhoto.filepath}</span>
              </div>

              {/* People Tagging */}
              {selectedPhoto.people && selectedPhoto.people.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                  <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                    People:
                  </span>
                  {selectedPhoto.people.map((face) => {
                    const isEditing = editingFaceId === face.face_id;

                    if (isEditing) {
                      return (
                        <div key={face.face_id} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <input
                            type="text"
                            autoFocus
                            placeholder="Name..."
                            value={faceInputName}
                            disabled={isSavingFace}
                            onChange={(e) => setFaceInputName(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter') handleSaveFaceName(face.face_id);
                              else if (e.key === 'Escape') setEditingFaceId(null);
                            }}
                            style={{
                              height: '22px',
                              padding: '0 6px',
                              backgroundColor: 'var(--bg-input)',
                              color: 'var(--text-primary)',
                              border: '1px solid var(--accent)',
                              borderRadius: '3px',
                              fontSize: '11px',
                            }}
                          />
                          <button
                            disabled={isSavingFace}
                            onClick={() => handleSaveFaceName(face.face_id)}
                            style={{
                              height: '22px',
                              padding: '0 8px',
                              backgroundColor: 'var(--accent)',
                              color: 'var(--text-on-accent)',
                              border: 'none',
                              borderRadius: '3px',
                              fontSize: '11px',
                            }}
                          >
                            Save
                          </button>
                        </div>
                      );
                    }

                    return (
                      <div key={face.face_id} style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span
                          onClick={() => {
                            setEditingFaceId(face.face_id);
                            setFaceInputName(face.is_named ? face.name : '');
                          }}
                          title="Click to rename"
                          style={{
                            fontSize: '11px',
                            padding: '2px 8px',
                            borderRadius: '3px',
                            backgroundColor: face.is_named ? 'var(--bg-selected-subtle)' : 'var(--bg-input)',
                            color: face.is_named ? 'var(--accent)' : 'var(--text-secondary)',
                            cursor: 'pointer',
                            fontWeight: face.is_named ? 600 : 400,
                          }}
                        >
                          👤 {face.name} ✎
                        </span>
                        {face.person_id && (
                          <button
                            onClick={() => handleRemoveFaceFromPerson(face.face_id, face.person_id)}
                            title="Remove this photo from this person"
                            style={{
                              fontSize: '10px',
                              padding: '2px 6px',
                              borderRadius: '3px',
                              backgroundColor: 'transparent',
                              color: '#ef4444',
                              border: '1px solid #ef4444',
                              cursor: 'pointer',
                            }}
                          >
                            ✕ Remove
                          </button>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
