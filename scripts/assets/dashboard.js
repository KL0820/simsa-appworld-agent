const $ = (id) => document.getElementById(id);
const titles = {
  bootstrap: "Load AppWorld task", plan: "Plan the task", retrieval: "Find relevant APIs",
  execution: "Execute with selected APIs", control: "Check progress",
  submission: "Submit the result", evaluation: "Evaluate outcome", error: "Run stopped"
};

function element(tag, className, value) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (value !== undefined && value !== null) node.textContent = String(value);
  return node;
}

function addText(parent, text) {
  if (text) parent.append(element("p", "", text));
}

function addChips(parent, label, values) {
  if (!Array.isArray(values) || !values.length) return;
  parent.append(element("div", "label", label));
  const row = element("div", "chips");
  for (const value of values) row.append(element("span", "chip", value));
  parent.append(row);
}

function planItemMarker(item) {
  const marker = element("li", "plan-item-marker");
  const label = `PLAN ITEM ${item.number} OF ${item.total}${item.cycle > 1 ? ` · CYCLE ${item.cycle}` : ""}`;
  marker.append(element("span", "label", label));
  marker.append(element("strong", "", `${item.app ? item.app + " · " : ""}${item.task || "Unnamed plan item"}`));
  return marker;
}

function eventBody(event) {
  const body = element("div", "event-body");
  if (event.stage_status === "running") {
    if (event.type === "retrieval" && event.plan_item?.task) addText(body, `Searching APIs for: ${event.plan_item.task}`);
    if (event.type === "execution" && event.plan_item?.task) addText(body, `Executing: ${event.plan_item.task}`);
    if (event.type === "control" && event.plan_item?.task) addText(body, `Checking progress on: ${event.plan_item.task}`);
    addText(body, "Working… this stage will update when the agent returns a result.");
    return body;
  }
  if (event.type === "plan") {
    const steps = Array.isArray(event.milestones) ? event.milestones : [];
    if (!steps.length) addText(body, "The agent did not provide a step list.");
    const list = element("ol");
    for (const step of steps) list.append(element("li", "", `${step.app ? step.app + " · " : ""}${step.task || "Unnamed step"}`));
    body.append(list);
  } else if (event.type === "retrieval") {
    if (event.plan_item?.task) addText(body, `APIs considered for: ${event.plan_item.task}`);
    addChips(body, "MATCHED APPS", event.apps);
    addChips(body, "CANDIDATE APIS", event.apis);
    if (event.apis?.length) addText(body, "Candidates considered here; the execution stage shows which APIs were actually called.");
    if (!event.apis?.length) addText(body, "No API names were returned.");
  } else if (event.type === "execution") {
    addText(body, event.summary || "Code ran in the AppWorld sandbox.");
    addChips(body, "APIS CALLED", event.called_apis);
    addText(body, event.milestone_done ? "The current step is complete." : "The current step needs more work.");
  } else if (event.type === "control") {
    addText(body, `Next action: ${event.action || "continue"}`);
    addText(body, event.reason);
  } else if (event.type === "evaluation") {
    const count = event.passed != null && event.total != null ? `${event.passed}/${event.total} checks passed.` : "No evaluator count available.";
    addText(body, `${count} ${event.passed_all ? "All checks passed." : ""}`.trim());
  } else if (event.type === "error") {
    addText(body, event.message || "Inspect the local run log for details.");
  } else if (event.type === "submission") {
    addText(body, event.status || "The agent submitted its final answer.");
  }
  return body;
}

function render(data) {
  const preview = data.mode === "preview";
  $("mode").textContent = preview ? "ILLUSTRATIVE PREVIEW" : (data.mode === "replay" ? "RECORDED LIVE RUN" : "LIVE MODEL RUN");
  $("mode").className = `mode ${preview ? "preview" : "live"}`;
  $("mode-note").textContent = preview
    ? "A synthetic walkthrough of the interface. No model or AppWorld task is run; use live mode to see real agent behavior."
    : (data.mode === "replay" ? "A completed, real AppWorld run recorded locally on this computer." : "Live stage updates appear about every 0.6 seconds as the agent works. Model tokens are not streamed. This page stays on your computer.");
  const task = data.events.find((event) => event.type === "task");
  if (task) {
    $("task-id").textContent = task.task_id || "";
    $("task-text").textContent = task.instruction || "No task text available.";
  }
  const last = data.events.at(-1);
  const finished = last && ["evaluation", "error", "preview_complete"].includes(last.type);
  $("run-status").replaceChildren();
  if (!finished) $("run-status").append(element("span", "spinner"));
  $("run-status").append(document.createTextNode(finished ? (last.type === "error" ? "Stopped" : "Finished") : "Agent working"));
  const timeline = $("timeline");
  timeline.replaceChildren();
  const stages = data.stages || [];
  let shownCycle = "";
  for (const event of stages) {
    if (event.plan_item) {
      const item = event.plan_item;
      const cycle = `${item.number}:${item.cycle}:${item.task}`;
      if (cycle !== shownCycle) timeline.append(planItemMarker(item));
      shownCycle = cycle;
    }
    const item = element("li", `event ${event.stage_status === "running" ? "running" : "complete"}`);
    const head = element("div", "event-head");
    head.append(element("strong", "", titles[event.type] || event.type));
    if (event.at) head.append(element("time", "", new Date(event.at * 1000).toLocaleTimeString()));
    item.append(head);
    if (event.type !== "bootstrap" || event.stage_status === "running") item.append(eventBody(event));
    timeline.append(item);
  }
  if (!stages.length) timeline.append(element("li", "event running", "Waiting for the agent to start…"));
}

async function refresh() {
  try {
    const response = await fetch("/api/events", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $("run-status").textContent = "Disconnected";
  }
}

refresh();
setInterval(refresh, 600);
