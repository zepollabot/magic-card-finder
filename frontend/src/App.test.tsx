import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((event: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  send(_data: string) {}

  close() {
    if (this.onclose) this.onclose();
  }
}

describe("App", () => {
  beforeEach(() => {
    // @ts-expect-error override
    global.WebSocket = MockWebSocket;
  });

  afterEach(() => {
    MockWebSocket.instances = [];
  });

  it("renders tabs and basic layout", () => {
    render(<App />);
    expect(screen.getByText(/Magic Card Finder/i)).toBeInTheDocument();
    expect(screen.getByText(/Listing URLs/i)).toBeInTheDocument();
    expect(screen.getByText(/Upload Images/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Card Names/i })).toBeInTheDocument();
  });

  it("drives status bar via WebSocket for card names", async () => {
    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /Card Names/i }));

    const textarea = screen.getByLabelText(/Card names/i);
    fireEvent.change(textarea, { target: { value: "Tarmogoyf" } });

    fireEvent.click(screen.getByText(/Analyze/i));

    const ws = MockWebSocket.instances[0];
    expect(ws.url).toContain("/api/ws/analyze");

    // @ts-expect-error manual trigger
    ws.onopen && ws.onopen();

    ws.onmessage &&
      ws.onmessage({
        data: JSON.stringify({
          type: "steps",
          steps: [
            { id: "scryfall_normalize", label: "Card normalization", index: 0 },
            { id: "price_source_scryfall", label: "Fetch prices from scryfall", index: 1 },
          ],
        }),
      });

    ws.onmessage &&
      ws.onmessage({
        data: JSON.stringify({
          type: "step_start",
          step_id: "scryfall_normalize",
          step_index: 0,
          message: "Resolving",
        }),
      });

    ws.onmessage &&
      ws.onmessage({
        data: JSON.stringify({
          type: "step_complete",
          step_id: "scryfall_normalize",
          step_index: 0,
        }),
      });

    ws.onmessage &&
      ws.onmessage({
        data: JSON.stringify({
          type: "step_start",
          step_id: "price_source_scryfall",
          step_index: 1,
          message: "Fetching prices",
        }),
      });

    ws.onmessage &&
      ws.onmessage({
        data: JSON.stringify({
          type: "step_complete",
          step_id: "price_source_scryfall",
          step_index: 1,
        }),
      });

    ws.onmessage &&
      ws.onmessage({
        data: JSON.stringify({
          type: "result",
          result: { analysis_id: "123", cards: [], price_sources: [] },
        }),
      });

    await waitFor(() => {
      expect(screen.getByText(/Analysis #123/i)).toBeInTheDocument();
      expect(screen.getByText(/Card normalization/i)).toBeInTheDocument();
      expect(screen.getByText(/Analysis complete/i)).toBeInTheDocument();
    });
  });
});
