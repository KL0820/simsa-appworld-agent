const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class FakeNode {
  constructor(tag = "div") {
    this.tag = tag;
    this.className = "";
    this.children = [];
    this.value = "";
  }

  set textContent(value) {
    this.value = String(value);
    this.children = [];
  }

  get textContent() {
    return this.value + this.children.map((child) => child.textContent).join(" ");
  }

  append(...children) {
    this.children.push(...children);
  }

  replaceChildren(...children) {
    this.children = [...children];
  }
}

const ids = [
  "mode", "mode-note", "task-id", "task-text", "run-status", "timeline",
  "metrics-section", "metric-duration", "metric-calls", "metric-attempts",
  "metric-tokens", "metric-prompt", "metric-completion", "metric-thoughts",
];
const nodes = Object.fromEntries(ids.map((id) => [id, new FakeNode()]));
const context = {
  document: {
    getElementById: (id) => nodes[id],
    createElement: (tag) => new FakeNode(tag),
    createTextNode: (text) => {
      const node = new FakeNode("text");
      node.textContent = text;
      return node;
    },
  },
  fetch: () => new Promise(() => {}),
  setInterval: () => {},
};
vm.createContext(context);
const source = fs.readFileSync(path.join(__dirname, "../scripts/assets/dashboard.js"), "utf8");
vm.runInContext(source, context);

const item = (number, total, task, cycle = 1) => ({ number, total, app: "spotify", task, cycle });
const completed = (type, planItem = undefined) => ({ type, stage_status: "complete", plan_item: planItem });
const stage = (value) => ({ kind: "stage", stage: value });
const descendants = (node, className) => node.children.flatMap((child) => [
  ...(child.className === className ? [child] : []), ...descendants(child, className),
]);
const group = (planItem, cycles) => ({
  kind: "plan_item", item: planItem,
  cycles: cycles.map(([number, stages]) => ({ number, item: item(planItem.number, planItem.total, planItem.task, number), stages })),
});

context.render({
  mode: "replay",
  events: [{ type: "task", task_id: "example", instruction: "Example" }, { type: "evaluation" }],
  blocks: [
    stage({ ...completed("plan"), milestones: [{ app: "spotify", task: "First" }, { app: "spotify", task: "Second" }] }),
    group(item(1, 2, "First"), [
      [1, [
        { ...completed("retrieval", item(1, 2, "First")), apis: ["spotify.search"] },
        completed("execution", item(1, 2, "First")),
        completed("control", item(1, 2, "First")),
      ]],
      [2, [{ ...completed("retrieval", item(1, 2, "First", 2)), apis: ["spotify.get_playlist"] }]],
    ]),
    group(item(2, 2, "Second"), [[1, [completed("retrieval", item(2, 2, "Second"))]]]),
  ],
});

const groups = descendants(nodes.timeline, "plan-group");
assert.equal(groups.length, 2);
assert.equal(nodes.timeline.children.length, 1);
assert.equal(descendants(nodes.timeline.children[0], "plan-list").length, 1);
assert.equal(descendants(nodes.timeline.children[0], "plan-cycles").length, 1);
assert.equal(descendants(nodes.timeline.children[0], "plan-items").length, 1);
assert.match(groups[0].textContent, /PLAN ITEM 1 OF 2/);
assert.equal(groups[0].children.filter((node) => node.className === "cycle").length, 2);
assert.match(groups[0].textContent, /Cycle 2/);
assert.match(groups[1].textContent, /PLAN ITEM 2 OF 2/);
assert.match(groups[0].textContent, /spotify.search/);
assert.doesNotMatch(groups[0].textContent, /APIs considered for: First/);

context.render({
  mode: "live",
  events: [{ type: "task", task_id: "example", instruction: "Example" }, { type: "phase_started", phase: "FIND" }],
  blocks: [
    stage({ ...completed("plan"), milestones: [{ app: "spotify", task: "First" }] }),
    group(item(1, 1, "First"), [[1, [
      { type: "retrieval", stage_status: "running", plan_item: item(1, 1, "First") },
    ]]]),
  ],
});
assert.match(nodes.timeline.textContent, /Find relevant APIs/);
assert.match(nodes.timeline.textContent, /Working…/);
assert.equal(descendants(nodes.timeline, "plan-list").length, 1);
assert.equal(descendants(nodes.timeline, "plan-cycles").length, 1);
assert.equal(descendants(nodes.timeline, "plan-group").length, 1);
assert.equal(nodes["metrics-section"].hidden, true);

context.render({
  mode: "replay",
  events: [{ type: "task", task_id: "example", instruction: "Example" }, { type: "metrics" }],
  blocks: [stage(completed("evaluation"))],
  metrics: {
    duration_s: 69.555, llm_calls: 8, llm_call_attempts: 9,
    prompt_tokens: 27555, completion_tokens: 669,
    thoughts_tokens: 2969, total_tokens: 31193,
  },
});
assert.equal(nodes["metrics-section"].hidden, false);
assert.equal(nodes["metric-duration"].textContent, "1m 9.6s");
assert.equal(nodes["metric-calls"].textContent, "8");
assert.equal(nodes["metric-attempts"].textContent, "9 attempts in total");
assert.equal(nodes["metric-attempts"].hidden, false);
assert.equal(nodes["metric-tokens"].textContent, "31,193");
assert.equal(nodes["metric-prompt"].textContent, "27,555");

context.renderMetrics({ llm_calls: 8, llm_call_attempts: 8 });
assert.equal(nodes["metric-attempts"].hidden, true);
assert.equal(nodes["metric-attempts"].textContent, "");

console.log("Dashboard plan-item grouping passed");
