import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { StatusBar, StepInfo, StepStatus } from "./StatusBar";

describe("StatusBar", () => {
  const steps: StepInfo[] = [
    { id: "step1", label: "Step 1", index: 0 },
    { id: "step2", label: "Step 2", index: 1 },
  ];

  it("renders nothing when idle with no steps", () => {
    const { container } = render(
      <StatusBar steps={[]} stepStatuses={{}} status="idle" message={null} subProgress={null} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows active step label and spinner when running", () => {
    const statuses: Record<string, StepStatus> = { step1: "active", step2: "pending" };
    render(
      <StatusBar
        steps={steps}
        stepStatuses={statuses}
        status="running"
        message="Working"
        subProgress={{ current: 1, total: 2 }}
      />
    );

    expect(screen.getAllByText(/Step 1/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/\(1\/2\)/)).toBeInTheDocument();
    expect(screen.getByText(/Working/)).toBeInTheDocument();
    expect(screen.getByText("0/2")).toBeInTheDocument();
  });

  it("shows completed step with checkmark styling", () => {
    const statuses: Record<string, StepStatus> = { step1: "completed", step2: "active" };
    const { container } = render(
      <StatusBar steps={steps} stepStatuses={statuses} status="running" message={null} subProgress={null} />
    );

    const pills = container.querySelectorAll(".rounded-pill");
    expect(pills.length).toBe(2);
    expect(pills[0].querySelector(".bi-check-lg")).toBeTruthy();
    expect(screen.getByText("1/2")).toBeInTheDocument();
  });

  it("shows completion text when done", () => {
    const statuses: Record<string, StepStatus> = { step1: "completed", step2: "completed" };
    render(
      <StatusBar steps={steps} stepStatuses={statuses} status="done" message={null} subProgress={null} />
    );
    expect(screen.getByText(/Analysis complete/i)).toBeInTheDocument();
    expect(screen.getByText("2/2")).toBeInTheDocument();
  });

  it("shows error state", () => {
    const statuses: Record<string, StepStatus> = { step1: "completed", step2: "active" };
    render(
      <StatusBar steps={steps} stepStatuses={statuses} status="error" message={null} subProgress={null} />
    );
    expect(screen.getByText(/Analysis failed/i)).toBeInTheDocument();
  });
});
