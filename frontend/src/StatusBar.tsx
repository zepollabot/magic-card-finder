export interface StepInfo {
  id: string;
  label: string;
  index: number;
}

export type StatusState = "idle" | "running" | "done" | "error";

export type StepStatus = "pending" | "active" | "completed";

interface StatusBarProps {
  steps: StepInfo[];
  stepStatuses: Record<string, StepStatus>;
  status: StatusState;
  message?: string | null;
  subProgress?: { current: number; total: number } | null;
}

export function StatusBar({ steps, stepStatuses, status, message, subProgress }: StatusBarProps) {
  if (!steps.length && status === "idle") return null;

  const completedCount = Object.values(stepStatuses).filter((s) => s === "completed").length;
  const total = steps.length || 1;
  const progressPercent = Math.min(100, Math.round((completedCount / total) * 100));

  const activeStep = steps.find((s) => stepStatuses[s.id] === "active");

  const borderColor =
    status === "error" ? "#dc3545" : status === "done" ? "#198754" : "#0d6efd";

  return (
    <div
      className="card shadow-sm mb-4"
      style={{ borderLeft: `3px solid ${borderColor}` }}
    >
      <div className="card-body py-3 px-4">
        <div className="d-flex align-items-center justify-content-between mb-2">
          <div className="d-flex align-items-center gap-2">
            {status === "running" && (
              <span className="spinner-border spinner-border-sm text-primary" role="status" />
            )}
            {status === "done" && <i className="bi bi-check-circle-fill text-success" />}
            {status === "error" && <i className="bi bi-x-circle-fill text-danger" />}
            <span className="fw-semibold" style={{ fontSize: "0.85rem" }}>
              {status === "running" && activeStep && (
                <>
                  {activeStep.label}
                  {subProgress && (
                    <span className="text-muted fw-normal ms-1">
                      ({subProgress.current}/{subProgress.total})
                    </span>
                  )}
                </>
              )}
              {status === "running" && !activeStep && "Starting analysis..."}
              {status === "done" && "Analysis complete"}
              {status === "error" && "Analysis failed"}
            </span>
          </div>
          <div className="d-flex align-items-center gap-2">
            {message && (
              <span className="text-muted" style={{ fontSize: "0.8rem" }}>
                {message}
              </span>
            )}
            <span className="text-muted" style={{ fontSize: "0.8rem" }}>
              {completedCount}/{total}
            </span>
          </div>
        </div>

        <div
          className="progress"
          style={{ height: "4px", borderRadius: "2px", backgroundColor: "#e9ecef" }}
        >
          <div
            className={`progress-bar ${status === "running" ? "progress-bar-animated" : ""}`}
            role="progressbar"
            aria-valuenow={progressPercent}
            aria-valuemin={0}
            aria-valuemax={100}
            style={{
              width: `${progressPercent}%`,
              transition: "width 0.4s ease",
              backgroundColor: borderColor,
            }}
          />
        </div>

        {steps.length > 0 && (
          <div className="d-flex flex-wrap gap-1 mt-2" style={{ fontSize: "0.78rem" }}>
            {steps.map((step) => {
              const ss = stepStatuses[step.id] || "pending";
              return (
                <span
                  key={step.id}
                  className="d-inline-flex align-items-center gap-1 px-2 py-1 rounded-pill"
                  style={{
                    border: `1px solid ${ss === "active" ? "#0d6efd" : ss === "completed" ? "#198754" : "#dee2e6"}`,
                    backgroundColor:
                      ss === "active" ? "#e7f1ff" : ss === "completed" ? "#d1e7dd" : "#f8f9fa",
                    color: ss === "active" ? "#0d6efd" : ss === "completed" ? "#0f5132" : "#6c757d",
                    transition: "all 0.25s ease",
                    whiteSpace: "nowrap",
                  }}
                >
                  {ss === "completed" && <i className="bi bi-check-lg" style={{ fontSize: "0.7rem" }} />}
                  {ss === "active" && (
                    <span
                      className="spinner-border text-primary"
                      style={{ width: "0.6rem", height: "0.6rem", borderWidth: "1.5px" }}
                    />
                  )}
                  {step.label}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
