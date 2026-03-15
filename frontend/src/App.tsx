import { ChangeEvent, FormEvent, useCallback, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { StatusBar, StepInfo, StatusState, StepStatus } from "./StatusBar";

interface CardPriceInfo {
  source: string;
  currency: string;
  price_low?: number | null;
  price_avg?: number | null;
  price_high?: number | null;
  trend_price?: number | null;
  set_name?: string | null;
  collector_number?: string | null;
  image_url?: string | null;
  thumbnail_url?: string | null;
}

interface CardReportItem {
  card_name: string;
  set_name?: string | null;
  collector_number?: string | null;
  image_url?: string | null;
  thumbnail_url?: string | null;
  prices: CardPriceInfo[];
}

interface AnalysisResponse {
  analysis_id: string;
  cards: CardReportItem[];
  price_sources?: string[];
}

/** One row in the report table: card + expansion, with price per source (average) */
interface ReportRow {
  card_name: string;
  set_name: string;
  collector_number: string;
  image_url?: string | null;
  thumbnail_url?: string | null;
  pricesBySource: Record<string, { price_avg: number; currency: string }>;
}

type ProgressEvent =
  | { type: "steps"; steps: StepInfo[] }
  | { type: "step_start" | "step_complete"; step_id: string; step_index: number; message?: string }
  | { type: "progress"; step_id: string; current: number; total: number }
  | { type: "result"; result: AnalysisResponse }
  | { type: "error"; message: string };

function buildReportRows(
  cards: CardReportItem[],
  enabledPriceSources?: string[]
): { rows: ReportRow[]; sources: string[] } {
  const sourceSet = new Set<string>();
  const rowMap = new Map<string, ReportRow>();

  for (const card of cards) {
    // Group this card's prices by set_name
    const bySet = new Map<string, CardPriceInfo[]>();
    for (const p of card.prices) {
      const setKey = p.set_name ?? "";
      if (!bySet.has(setKey)) bySet.set(setKey, []);
      bySet.get(setKey)!.push(p);
      if (p.source && p.source !== "__noprice__") {
        sourceSet.add(p.source);
      }
    }

    for (const [set_name, prices] of bySet) {
      const first = prices[0];
      const collector_number = first?.collector_number ?? "";
      const rowKey = `${card.card_name}|${set_name}`;
      const pricesBySource: Record<string, { price_avg: number; currency: string }> = {};
      for (const p of prices) {
        const avg = p.price_avg ?? 0;
        if (p.source && p.source !== "__noprice__" && (avg > 0 || p.price_avg === 0)) {
          pricesBySource[p.source] = { price_avg: avg, currency: p.currency };
        }
      }
      rowMap.set(rowKey, {
        card_name: card.card_name,
        set_name,
        collector_number,
        image_url: first?.image_url ?? card.image_url,
        thumbnail_url: first?.thumbnail_url ?? first?.image_url ?? card.thumbnail_url ?? card.image_url,
        pricesBySource,
      });
    }
  }

  // Use enabled price sources from API when present so all configured columns show (e.g. cardtrader)
  const sourcesFromData = Array.from(sourceSet).sort();
  const sources =
    (enabledPriceSources?.length ?? 0) > 0
      ? [...new Set([...enabledPriceSources, ...sourcesFromData])].sort()
      : sourcesFromData;

  const rows = Array.from(rowMap.values()).sort(
    (a, b) => a.card_name.localeCompare(b.card_name) || a.set_name.localeCompare(b.set_name)
  );
  return { rows, sources };
}

function App() {
  const [tab, setTab] = useState(0);
  const [urls, setUrls] = useState("");
  const [files, setFiles] = useState<FileList | null>(null);
  const [cardNames, setCardNames] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [report, setReport] = useState<AnalysisResponse | null>(null);
  const [openCards, setOpenCards] = useState<Record<string, boolean>>({});
  const [steps, setSteps] = useState<StepInfo[]>([]);
  const [stepStatuses, setStepStatuses] = useState<Record<string, StepStatus>>({});
  const [statusState, setStatusState] = useState<StatusState>("idle");
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [subProgress, setSubProgress] = useState<{ current: number; total: number } | null>(null);
  const [thumbnailZoom, setThumbnailZoom] = useState<{ url: string; x: number; y: number } | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const markStep = useCallback((stepId: string, status: StepStatus) => {
    setStepStatuses((prev) => ({ ...prev, [stepId]: status }));
  }, []);

  const { rows, sources } = useMemo(() => {
    if (!report?.cards?.length) return { rows: [] as ReportRow[], sources: [] as string[] };
    return buildReportRows(report.cards, report.price_sources);
  }, [report]);

  const rowsByCard = useMemo(() => {
    const map = new Map<string, ReportRow[]>();
    for (const row of rows) {
      if (!map.has(row.card_name)) map.set(row.card_name, []);
      map.get(row.card_name)!.push(row);
    }
    return map;
  }, [rows]);

  const toggleCardOpen = (cardName: string) => {
    setOpenCards((prev) => ({ ...prev, [cardName]: !(prev[cardName] ?? true) }));
  };

  const handleTabChange = (idx: number) => () => setTab(idx);
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => setFiles(e.target.files);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setReport(null);
    setSteps([]);
    setStepStatuses({});
    setStatusState("idle");
    setStatusMessage(null);
    setSubProgress(null);

    const resolveFeature = () => {
      if (tab === 0) return "card_names";
      if (tab === 1) return "upload_images";
      return "scrape_url";
    };

    const feature = resolveFeature();

    const names = cardNames
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean);
    const urlLines = urls
      .split("\n")
      .map((u) => u.trim())
      .filter(Boolean);

    const readFileAsBase64 = (file: File) =>
      new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error("Failed to read file"));
        reader.onload = () => {
          const result = reader.result;
          if (typeof result === "string") {
            const commaIndex = result.indexOf(",");
            resolve(commaIndex >= 0 ? result.slice(commaIndex + 1) : result);
          } else {
            reject(new Error("Unexpected file reader result"));
          }
        };
        reader.readAsDataURL(file);
      });

    try {
      let filesBase64: string[] = [];
      if (files && files.length > 0) {
        filesBase64 = await Promise.all(Array.from(files).map((f) => readFileAsBase64(f)));
      }

      const payload: Record<string, unknown> = {};
      if (feature === "card_names") {
        payload.names = names.length ? names : urlLines;
      } else {
        payload.urls = urlLines;
        payload.files = filesBase64;
      }

      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      const wsUrl = `${protocol}://${window.location.host}/api/ws/analyze`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      setStatusState("running");

      ws.onopen = () => {
        ws.send(JSON.stringify({ feature, payload }));
      };

      ws.onmessage = (event: MessageEvent) => {
        let parsed: ProgressEvent;
        try {
          parsed = JSON.parse(event.data as string) as ProgressEvent;
        } catch {
          return;
        }

        if (parsed.type === "steps") {
          setSteps(parsed.steps);
          const initial: Record<string, StepStatus> = {};
          for (const s of parsed.steps) initial[s.id] = "pending";
          setStepStatuses(initial);
          return;
        }

        if (parsed.type === "step_start") {
          markStep(parsed.step_id, "active");
          setStatusState("running");
          if (parsed.message) {
            setStatusMessage(parsed.message);
          }
          return;
        }

        if (parsed.type === "step_complete") {
          markStep(parsed.step_id, "completed");
          return;
        }

        if (parsed.type === "progress") {
          setSubProgress({ current: parsed.current, total: parsed.total });
          return;
        }

        if (parsed.type === "error") {
          setError(parsed.message);
          setStatusState("error");
          setLoading(false);
          ws.close();
          return;
        }

        if (parsed.type === "result") {
          setReport(parsed.result);
          setStatusState("done");
          setLoading(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        setError("WebSocket error");
        setStatusState("error");
        setLoading(false);
      };

      ws.onclose = () => {
        wsRef.current = null;
      };
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unexpected error");
      setStatusState("error");
      setLoading(false);
    }
  };

  const downloadCsv = async () => {
    if (!report) return;
    const resp = await fetch(`/api/report/${report.analysis_id}/csv`);
    const text = await resp.text();
    const blob = new Blob([text], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `analysis-${report.analysis_id}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-vh-100 bg-light">
      <nav className="navbar navbar-expand-lg navbar-dark bg-primary shadow-sm">
        <div className="container">
          <span className="navbar-brand fw-bold">
            <i className="bi bi-search me-2" />
            Magic Card Finder
          </span>
        </div>
      </nav>

      <main className="container py-4">
        <div className="card shadow-sm border-0 mb-4">
          <div className="card-body p-4">
            <ul className="nav nav-tabs nav-fill mb-4" role="tablist">
              <li className="nav-item">
                <button
                  type="button"
                  className={`nav-link ${tab === 0 ? "active" : ""}`}
                  onClick={handleTabChange(0)}
                >
                  <i className="bi bi-card-text me-1" /> Card Names
                </button>
              </li>
              <li className="nav-item">
                <button
                  type="button"
                  className={`nav-link ${tab === 1 ? "active" : ""}`}
                  onClick={handleTabChange(1)}
                >
                  <i className="bi bi-image me-1" /> Upload Images
                </button>
              </li>
              <li className="nav-item">
                <button
                  type="button"
                  className={`nav-link ${tab === 2 ? "active" : ""}`}
                  onClick={handleTabChange(2)}
                >
                  <i className="bi bi-link-45deg me-1" /> Listing URLs
                </button>
              </li>
            </ul>

            <form onSubmit={handleSubmit}>
              {tab === 0 && (
                <div className="mb-3">
                  <label className="form-label fw-semibold" htmlFor="card-names-input">
                    Card names (one per line; optional set: &quot;Card Name, Set Name&quot;)
                  </label>
                  <textarea
                    id="card-names-input"
                    className="form-control font-monospace"
                    rows={4}
                    value={cardNames}
                    onChange={(e) => setCardNames(e.target.value)}
                    placeholder={"Llanowar Elves\nLightning Bolt, Dominaria"}
                  />
                </div>
              )}
              {tab === 1 && (
                <div className="mb-3">
                  <label className="form-label fw-semibold" htmlFor="upload-images-input">
                    Select images
                  </label>
                  <div>
                    <input
                      id="upload-images-input"
                      type="file"
                      className="form-control"
                      accept="image/*"
                      multiple
                      onChange={handleFileChange}
                    />
                    {files && (
                      <small className="text-muted d-block mt-1">{files.length} file(s) selected</small>
                    )}
                  </div>
                </div>
              )}
              {tab === 2 && (
                <div className="mb-3">
                  <label className="form-label fw-semibold" htmlFor="listing-urls-input">
                    Listing URLs (one per line)
                  </label>
                  <textarea
                    id="listing-urls-input"
                    className="form-control font-monospace"
                    rows={4}
                    value={urls}
                    onChange={(e) => setUrls(e.target.value)}
                    placeholder="https://..."
                  />
                </div>
              )}

              <div className="d-flex flex-wrap gap-2 align-items-center">
                <button type="submit" className="btn btn-primary" disabled={loading}>
                  {loading ? (
                    <>
                      <span className="spinner-border spinner-border-sm me-1" role="status" />
                      Analyze
                    </>
                  ) : (
                    <>
                      <i className="bi bi-cpu me-1" /> Analyze
                    </>
                  )}
                </button>
                {report && (
                  <button type="button" className="btn btn-outline-secondary" onClick={downloadCsv}>
                    <i className="bi bi-download me-1" /> Download CSV
                  </button>
                )}
              </div>

              {loading && (
                <div className="mt-3 small text-muted">Running analysis...</div>
              )}
              {error && (
                <div className="alert alert-danger mt-3 mb-0" role="alert">
                  <i className="bi bi-exclamation-triangle me-2" /> {error}
                </div>
              )}
            </form>
          </div>
        </div>

        {report && (
          <div className="card shadow-sm border-0">
            <div className="card-header bg-white py-3">
              <h5 className="card-title mb-0 fw-semibold">
                <i className="bi bi-table me-2" />
                Analysis #{report.analysis_id}
              </h5>
            </div>
            <div className="card-body p-0">
              {rows.length === 0 ? (
                <p className="text-muted p-4 mb-0">No cards with pricing data.</p>
              ) : (
                <div className="accordion" id="analysis-accordion">
                  {Array.from(rowsByCard.entries()).map(([cardName, cardRows], idx) => {
                    const isOpen = openCards[cardName] ?? true;
                    return (
                      <div className="accordion-item border-0 border-top" key={cardName}>
                        <h2 className="accordion-header" id={`heading-${idx}`}>
                          <button
                            type="button"
                            className={`accordion-button py-2 px-3 ${isOpen ? "" : "collapsed"}`}
                            onClick={() => toggleCardOpen(cardName)}
                          >
                            <span className="fw-semibold me-2">{cardName}</span>
                            <span className="badge bg-light text-muted">
                              {cardRows.length} expansion{cardRows.length !== 1 ? "s" : ""}
                            </span>
                          </button>
                        </h2>
                        <div
                          className={`accordion-collapse collapse ${isOpen ? "show" : ""}`}
                          aria-labelledby={`heading-${idx}`}
                        >
                          <div className="accordion-body p-0">
                            <div className="table-responsive">
                              <table className="table table-hover table-striped align-middle mb-0">
                                <thead className="table-light">
                                  <tr>
                                    <th className="fw-semibold">Card</th>
                                    <th className="fw-semibold">Set name</th>
                                    <th className="fw-semibold">Collector #</th>
                                    {sources.map((src) => (
                                      <th key={src} className="fw-semibold text-nowrap text-capitalize">
                                        {src}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {cardRows.map((row, rIdx) => (
                                    <tr key={`${cardName}-${row.set_name}-${rIdx}`}>
                                      <td>
                                        {row.thumbnail_url || row.image_url ? (
                                          <div
                                            className="d-inline-block thumbnail-wrapper"
                                            onMouseEnter={(e) => {
                                              if (!row.image_url) return;
                                              const rect = e.currentTarget.getBoundingClientRect();
                                              const pad = 8;
                                              const maxDim = 480; // 1.5x of previous 320
                                              let x = rect.right + pad;
                                              if (x + maxDim > window.innerWidth - pad) x = rect.left - maxDim - pad;
                                              if (x < pad) x = pad;
                                              let y = rect.top;
                                              if (y + maxDim > window.innerHeight - pad) y = window.innerHeight - maxDim - pad;
                                              if (y < pad) y = pad;
                                              setThumbnailZoom({ url: row.image_url, x, y });
                                            }}
                                            onMouseLeave={() => setThumbnailZoom(null)}
                                          >
                                            <img
                                              src={row.thumbnail_url || row.image_url!}
                                              alt={row.card_name}
                                              className="img-thumbnail"
                                              style={{ maxHeight: 64, maxWidth: 64 }}
                                            />
                                          </div>
                                        ) : (
                                          "—"
                                        )}
                                      </td>
                                      <td>{row.set_name || "—"}</td>
                                      <td className="text-muted">{row.collector_number || "—"}</td>
                                      {sources.map((src) => {
                                        const p = row.pricesBySource[src];
                                        if (!p)
                                          return (
                                            <td
                                              key={src}
                                              className="text-muted"
                                              title={`No data from ${src} for this expansion`}
                                            >
                                              —
                                            </td>
                                          );
                                        return (
                                          <td key={src} className="text-nowrap">
                                            {p.price_avg.toFixed(2)} ({p.currency})
                                          </td>
                                        );
                                      })}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </main>

      <StatusBar
        steps={steps}
        stepStatuses={stepStatuses}
        status={statusState}
        message={statusMessage}
        subProgress={subProgress}
      />

      {thumbnailZoom &&
        createPortal(
          <div
            className="border bg-white shadow-lg rounded overflow-hidden"
            style={{
              position: "fixed",
              left: thumbnailZoom.x,
              top: thumbnailZoom.y,
              zIndex: 1070,
              maxWidth: 480,
              maxHeight: 480,
              pointerEvents: "none",
            }}
          >
            <img
              src={thumbnailZoom.url}
              alt=""
              style={{ maxWidth: 480, maxHeight: 480, display: "block" }}
            />
          </div>,
          document.body
        )}
    </div>
  );
}

export default App;
