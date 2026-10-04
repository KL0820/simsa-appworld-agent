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

function formatCount(value) {
  return Number.isInteger(value) && value >= 0 ? value.toLocaleString("en-US") : "—";
}

function formatDuration(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "—";
  const minutes = Math.floor(seconds / 60);
  const remainder = (seconds % 60).toFixed(1);
  return minutes ? `${minutes}m ${remainder}s` : `${remainder}s`;
}

function renderMetrics(metrics) {
  $("metrics-section").hidden = !metrics;
  if (!metrics) return;
  $("metric-duration").textContent = formatDuration(metrics.duration_s);
  $("metric-calls").textContent = formatCount(metrics.llm_calls);
  $("metric-attempts").textContent = Number.isInteger(metrics.llm_call_attempts)
    ? `${formatCount(metrics.llm_call_attempts)} attempts` : "Attempts unavailable";
  $("metric-tokens").textContent = formatCount(metrics.total_tokens);
  $("metric-prompt").textContent = formatCount(metrics.prompt_tokens);
  $("metric-completion").textContent = formatCount(metrics.completion_tokens);
  $("metric-thoughts").textContent = formatCount(metrics.thoughts_tokens);
}

function eventBody(event) {
  const body = element("div", "event-body");
  if (event.stage_status === "running") {
    addText(body, "Working… this stage will update when the agent returns a result.");
    return body;
  }
  if (event.type === "plan") {
    const steps = Array.isArray(event.milestones) ? event.milestones : [];
    if (!steps.length) addText(body, "The agent did not provide a step list.");
    if (steps.length === 1) addText(body, "One plan item created. Its work is grouped below.");
    if (steps.length > 1) {
      const list = element("ol");
      for (const step of steps) list.append(element("li", "", `${step.app ? step.app + " · " : ""}${step.task || "Unnamed step"}`));
      body.append(list);
    }
  } else if (event.type === "retrieval") {
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

function renderStage(event, nested = false) {
  const classes = `event ${nested ? "cycle-event " : ""}${event.stage_status === "running" ? "running" : "complete"}`;
  const item = element("li", classes);
  const head = element("div", "event-head");
  head.append(element("strong", "", titles[event.type] || event.type));
  if (event.at) head.append(element("time", "", new Date(event.at * 1000).toLocaleTimeString()));
  item.append(head);
  if (event.type !== "bootstrap" || event.stage_status === "running") item.append(eventBody(event));
  return item;
}

function renderPlanGroup(block) {
  const group = element("li", "plan-group");
  const header = element("div", "plan-group-head");
  header.append(element("span", "label", `PLAN ITEM ${block.item.number} OF ${block.item.total}`));
  header.append(element("strong", "", `${block.item.app ? block.item.app + " · " : ""}${block.item.task || "Unnamed plan item"}`));
  group.append(header);
  for (const cycle of block.cycles) {
    const round = element("section", "cycle");
    const cycleHead = element("div", "cycle-head");
    cycleHead.append(element("span", "cycle-number", `Cycle ${cycle.number}`));
    if (cycle.item.task !== block.item.task) cycleHead.append(element("span", "cycle-revision", `Revised goal: ${cycle.item.task}`));
    round.append(cycleHead);
    const events = element("ol", "cycle-events");
    for (const stage of cycle.stages) events.append(renderStage(stage, true));
    round.append(events);
    group.append(round);
  }
  return group;
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
  const finished = last && ["evaluation", "metrics", "error", "preview_complete"].includes(last.type);
  $("run-status").replaceChildren();
  if (!finished) $("run-status").append(element("span", "spinner"));
  $("run-status").append(document.createTextNode(finished ? (last.type === "error" ? "Stopped" : "Finished") : "Agent working"));
  const timeline = $("timeline");
  timeline.replaceChildren();
  const blocks = data.blocks || [];
  let planItems = null;
  for (const block of blocks) {
    if (block.kind === "plan_item") {
      (planItems || timeline).append(renderPlanGroup(block));
      continue;
    }
    const stage = renderStage(block.stage);
    timeline.append(stage);
    planItems = null;
    if (block.stage.type === "plan") {
      stage.className += " plan-workflow";
      planItems = element("ol", "plan-items");
      stage.append(planItems);
    }
  }
  if (!blocks.length) timeline.append(element("li", "event running", "Waiting for the agent to start…"));
  renderMetrics(data.metrics);
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
