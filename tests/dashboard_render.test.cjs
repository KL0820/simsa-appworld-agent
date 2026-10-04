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

const ids = ["mode", "mode-note", "task-id", "task-text", "run-status", "timeline"];
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

context.render({
  mode: "replay",
  events: [{ type: "task", task_id: "example", instruction: "Example" }, { type: "evaluation" }],
  stages: [
    { ...completed("plan"), milestones: [{ app: "spotify", task: "First" }, { app: "spotify", task: "Second" }] },
    { ...completed("retrieval", item(1, 2, "First")), apis: ["spotify.search"] },
    completed("execution", item(1, 2, "First")),
    completed("control", item(1, 2, "First")),
    { ...completed("retrieval", item(1, 2, "First", 2)), apis: ["spotify.get_playlist"] },
    completed("retrieval", item(2, 2, "Second")),
  ],
});

const markers = nodes.timeline.children.filter((node) => node.className === "plan-item-marker");
assert.equal(markers.length, 3);
assert.match(markers[0].textContent, /PLAN ITEM 1 OF 2/);
assert.match(markers[1].textContent, /CYCLE 2/);
assert.match(markers[2].textContent, /PLAN ITEM 2 OF 2/);
assert.match(nodes.timeline.textContent, /APIs considered for: First/);
assert.doesNotMatch(nodes.timeline.textContent, /STEP 1/);

context.render({
  mode: "live",
  events: [{ type: "task", task_id: "example", instruction: "Example" }, { type: "phase_started", phase: "FIND" }],
  stages: [
    { ...completed("plan"), milestones: [{ app: "spotify", task: "First" }] },
    { type: "retrieval", stage_status: "running", plan_item: item(1, 1, "First") },
  ],
});
assert.match(nodes.timeline.textContent, /Searching APIs for: First/);
assert.equal(nodes.timeline.children.filter((node) => node.className === "plan-item-marker").length, 1);

console.log("Dashboard plan-item grouping passed");
