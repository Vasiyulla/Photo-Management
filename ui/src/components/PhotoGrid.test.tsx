import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PhotoGrid from './PhotoGrid';

describe('PhotoGrid component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders filter bar and photo gallery', async () => {
    const mockPhotos = [
      {
        id: 1,
        filepath: '/photos/sunset.jpg',
        date_taken: '2024-06-15T18:30:00',
        phash: 'hash1',
        processing_state: 'completed',
        created_at: '2024-06-15T18:30:00',
      },
      {
        id: 2,
        filepath: '/photos/beach.jpg',
        date_taken: '2024-07-20T11:00:00',
        phash: 'hash2',
        processing_state: 'completed',
        created_at: '2024-07-20T11:00:00',
        score: 0.89,
      },
    ];

    const mockPeople = [
      { id: 1, display_name: 'Alice', face_count: 3, cover_photo_id: 1, cover_bbox: null },
    ];

    globalThis.fetch = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/persons')) {
        return Promise.resolve({ ok: true, json: async () => mockPeople } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => mockPhotos } as Response);
    });

    render(<PhotoGrid apiBase="http://127.0.0.1:8000" />);

    expect(screen.getByPlaceholderText(/search photos/i)).toBeDefined();
    expect(screen.getByText(/filter by person/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText(/sunset\.jpg/i)).toBeDefined();
      expect(screen.getByText(/beach\.jpg/i)).toBeDefined();
      expect(screen.getByText(/89% match/i)).toBeDefined();
    });
  });

  it('filters by semantic text search query', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/persons')) {
        return Promise.resolve({ ok: true, json: async () => [] } as Response);
      }
      return Promise.resolve({
        ok: true,
        json: async () => [
          {
            id: 2,
            filepath: '/photos/beach.jpg',
            date_taken: '2024-07-20T11:00:00',
            phash: 'hash2',
            processing_state: 'completed',
            score: 0.95,
          },
        ],
      } as Response);
    });

    globalThis.fetch = fetchMock;

    render(<PhotoGrid apiBase="http://127.0.0.1:8000" />);

    const searchInput = screen.getByPlaceholderText(/search photos/i);
    fireEvent.change(searchInput, { target: { value: 'sunny beach' } });
    fireEvent.keyDown(searchInput, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('text_query=sunny+beach')
      );
      expect(screen.getByText(/beach\.jpg/i)).toBeDefined();
    });
  });

  it('filters by selected person', async () => {
    const mockPeople = [
      { id: 1, display_name: 'Alice', face_count: 5, cover_photo_id: 1, cover_bbox: null },
    ];

    const fetchMock = vi.fn().mockImplementation((url: string) => {
      if (url.includes('/api/persons')) {
        return Promise.resolve({ ok: true, json: async () => mockPeople } as Response);
      }
      return Promise.resolve({ ok: true, json: async () => [] } as Response);
    });

    globalThis.fetch = fetchMock;

    render(<PhotoGrid apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/Alice/)).toBeDefined();
    });

    const select = screen.getByTestId('person-filter-select');
    fireEvent.change(select, { target: { value: '1' } });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('person_id=1')
      );
    });
  });
});
