import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import PeopleGrid from './PeopleGrid';

describe('PeopleGrid component', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('renders loading state initially then displays people cards', async () => {
    const mockPeople = [
      {
        id: 1,
        display_name: 'Alice',
        face_count: 5,
        cover_photo_id: 10,
        cover_bbox: '10,10,50,50',
      },
      {
        id: 2,
        display_name: null,
        face_count: 2,
        cover_photo_id: 20,
        cover_bbox: '20,20,60,60',
      },
    ];

    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockPeople,
    } as Response);

    render(<PeopleGrid apiBase="http://127.0.0.1:8000" />);

    expect(screen.getByText(/loading people/i)).toBeDefined();

    await waitFor(() => {
      expect(screen.getByText('Alice')).toBeDefined();
      expect(screen.getByText('Unnamed Person')).toBeDefined();
      expect(screen.getByText('5 photos')).toBeDefined();
      expect(screen.getByText('2 photos')).toBeDefined();
    });
  });

  it('allows inline tap-to-rename and submits rename request', async () => {
    const mockPeople = [
      {
        id: 2,
        display_name: null,
        face_count: 3,
        cover_photo_id: 15,
        cover_bbox: null,
      },
    ];

    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => mockPeople,
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 2, display_name: 'Bob' }),
      } as Response);

    globalThis.fetch = fetchMock;

    render(<PeopleGrid apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText('Unnamed Person')).toBeDefined();
    });

    // Click to rename
    fireEvent.click(screen.getByText('Unnamed Person'));

    const input = screen.getByRole('textbox');
    expect(input).toBeDefined();

    // Type new name
    fireEvent.change(input, { target: { value: 'Bob' } });

    // Submit by pressing Enter
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'http://127.0.0.1:8000/api/persons/2/rename',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ name: 'Bob' }),
        })
      );
      expect(screen.getByText('Bob')).toBeDefined();
    });
  });

  it('displays empty state when no people exist', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => [],
    } as Response);

    render(<PeopleGrid apiBase="http://127.0.0.1:8000" />);

    await waitFor(() => {
      expect(screen.getByText(/no people detected/i)).toBeDefined();
    });
  });
});
