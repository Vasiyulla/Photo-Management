import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import DuplicatesReview from './DuplicatesReview';

describe('DuplicatesReview component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders duplicate pairs and resolves duplicate on keep left', async () => {
    const mockPairs = [
      {
        match_type: 'phash_exact',
        similarity: 1.0,
        photo_a: {
          id: 1,
          filepath: '/photos/orig.jpg',
          date_taken: '2023-01-01T12:00:00',
        },
        photo_b: {
          id: 2,
          filepath: '/photos/copy.jpg',
          date_taken: '2023-01-02T12:00:00',
        },
      },
    ];

    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPairs,
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok' }),
      } as Response);

    globalThis.fetch = fetchMock;

    render(<DuplicatesReview apiBase="http://127.0.0.1:8000" />);

    expect(screen.getByText(/loading duplicates/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getAllByText(/orig\.jpg/i).length).toBeGreaterThan(0);
      expect(screen.getAllByText(/copy\.jpg/i).length).toBeGreaterThan(0);
      expect(screen.getByText(/exact duplicate/i)).toBeDefined();
    });

    // Click "Keep Left" (marking photo 2 as duplicate)
    const keepLeftBtn = screen.getByRole('button', { name: /keep left/i });
    fireEvent.click(keepLeftBtn);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/duplicates/resolve',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ duplicate_photo_id: 2, action: 'mark_duplicate' }),
        })
      );
      // Queue is now empty
      expect(screen.getByText(/no duplicates/i)).toBeDefined();
    });
  });

  it('renders empty queue state when no duplicates exist', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    } as Response);

    render(<DuplicatesReview apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/no duplicates/i)).toBeDefined();
    });
  });
});
