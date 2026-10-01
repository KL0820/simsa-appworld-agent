---
topic: Canonical trajectory schema — 餵 viewer + 蒸餾 + reproducibility 的單一真相格式
date: 2026-06-05
basis: MIND-Skill(2605.08670,親讀 §3.1 τ 定義)+ adk-appworld-agent 多 subagent 實際 run(2a163ab_1 oracle 親跑)
design_for: (1) skill induction 輸入 (2) view builder (3) 論文 reproducibility appendix
status: Claude 草稿（schema 設計）。待 user 審。
user_reviewed:
reviewer_verdict:
reviewer_notes: |
---

# Canonical Trajectory Schema v1

> ⚠️ **2026-06-06 升級為「完整多輪 `turns[]`」(user 拍板)。** 權威設計見 `night-runs/notes/handoff/single_trajectory_source_spec.md` §1.5。
>
> 本文以下把 step 寫成單通 `input / output / result` —— **那是舊版、只放首通快照（`len 1`）**，會漏掉 executor 的 submit_final 輪與每個 repair 輪（= 模型看到 stdout/錯誤後那通），破壞「觀察→動作」配對。**正確版**：step 改帶 `turns[]`，每通 model 呼叫一筆：
> ```
> step: { …定位/狀態…, subagent_input, injected_skills,
>   turns: [ { request:{system_ref, context:[{role, text|function_call|function_response}], tools, tool_config},
>              response:{parts, finish_reason}, metrics:{*_tokens} }, … ],  # turn2+ 含 function_response = 完整多輪
>   output:{parsed, code, tool_calls, …(可 derive 自 turns)}, result, metrics }
> ```
> 來源 = ADK before/after_model_callback（**每通 fire**）→ 完整多輪自然落地、非額外工。下方 `input/output/result` 一律讀作「`turns[0]` 的對應欄」，正式以 turns[] 為準。

## 0. 三條設計原則

1. **保留「情境 → 決策 → 觀察」配對，不砍輸入。** 對齊直接競品 MIND-Skill（§3.1：`τ = {(thought_m, code_m, observation_m)}`，observation 即環境回傳＝下一步的輸入）。砍掉輸入就不是 trajectory，蒸餾也看不出「為何這樣選」。
2. **一份來源，多個消費者用 adapter。** 來源只存這份完整 trajectory；viewer / 蒸餾 / reproducibility 各用投影讀。不再養第二套來源格式。
3. **完整 ≠ 肥。** 固定不變的 system prompt（agent 的通用規則 + few-shot）只存一次、用 hash reference；每步只逐字留「變動的輸入情境」。

## 1. MIND-Skill 的 tuple → 你的多 subagent 對應

| MIND-Skill (monolithic ReAct) | 你的多 subagent agent |
|---|---|
| `thought_m`（推理） | `output.rationale`（plan 的 thoughts、continuation 的 rationale；executor 的推理內含於 code） |
| `code_m`（動作） | `output.parsed`（milestones / 選的 API / code_plan / code / submit / 續行決策——依 step 角色而異） |
| `observation_m`（環境回傳） | `result`（執行 step 的 sandbox stdout + api_trace；純決策 step 為 null）。同時，前一步的 observation 已被 prompt builder 串進**下一步的 `input.context`**（逐字） |

**關鍵差異（也是你的論文差異化）**：MIND-Skill 的 step 是同質的 `(thought, code, observation)`；你的是**異質** step 序列——planning / finding / continuation 是純決策（無環境觀察），只有 execute step 產生觀察。這個異質結構本身就編碼了「skill 掛在多 subagent pipeline 哪一層」這件 MIND-Skill 沒有的事。

## 2. 完整 schema

```jsonc
{
  "schema_version": "1.0",

  // ── 來源 / 可重現性（provenance）──
  "run_meta": {
    "producer": "oracle-claude" | "gemini-live",   // 誰跑的：oracle 蒸餾用 / 正常 baseline
    "model": "gemini-2.5-flash",                    // 執行模型（oracle 時=claude-*）
    "agent_commit": "<git sha of adk-appworld-agent>",
    "adk_version": "1.31.0",
    "rpc_url": "tcp://127.0.0.1:4243",
    "created_at": "<iso, 跑完後 stamp>",
    "impls": { "PLAN": "rough", "FIND": "community", "EXECUTE": "code_plan_execute" }
  },

  // ── 問題（the task）──
  "task": {
    "task_id": "2a163ab_1",
    "split": "train",                               // train / dev / test_*
    "instruction": "Like all the venmo transactions from today involving any of my roommates on my venmo social feed.",
    "datetime": "2023-05-18T21:26:01",
    "user_profile": { "first_name": "Melissa", "last_name": "Bailey", "email": "mel.bailey@gmail.com", "phone_number": "3383946795" }
  },

  // ── 設定（reproducibility pin）──
  "config": {
    "sampling": { "temperature": 0, "top_p": 1, "top_k": 1, "seed": 123, "max_output_tokens": 0 },
    "catalog_version": "<community/API catalog 的版本或 hash>",   // 釘住檢索母體
    "prompt_set_version": "<prompt templates 的 commit/hash>"     // 釘住 prompt 版本
  },

  // ── 結果（the outcome）──
  "result": {
    "evaluator": { "passed": 6, "failed": 0, "total": 6, "cleared": true },
    "final_answer": "null",                         // 提交的答案（action 題為 "null"）
    "status": "SUBMITTED"
  },

  // ── 便利投影：分解計畫（= steps[PLAN].output.parsed，頂層放一份好查）──
  "plan": {
    "thoughts": "...",
    "milestones": [
      { "index": 0, "app": "phone", "intent": "Read my roommates from phone contacts." },
      { "index": 1, "app": "venmo", "intent": "On my venmo social feed, ... like each such transaction." }
    ]
  },

  // ── 固定 system prompt 去重存放（每個 distinct prompt 存一次，step 用 hash 引用）──
  "system_prompts": {
    "096c30ce": { "agent": "rough_planner",                  "chars": 6381,  "text": "You are RoughPlanner ..." },
    "423c7c37": { "agent": "community_finder:community_select","chars": 2791,  "text": "You are an API routing agent ..." },
    "8ae34c51": { "agent": "community_finder:seed_filter",   "chars": 4114,  "text": "You are an API selector ..." },
    "cbcc1d73": { "agent": "code_planner",                   "chars": 15628, "text": "You are the AppWorld code planner ..." },
    "54daf68f": { "agent": "code_executor",                  "chars": 7176,  "text": "You are the AppWorld code executor ..." },
    "4c91b47d": { "agent": "continuation",                   "chars": 11249, "text": "You are a recovery controller ..." }
  },

  // ── 軌跡本體：有序的 (情境 → 決策 → 觀察) 序列 ──
  "steps": [
    {
      "step_id": 1,                 // 全域遞增序號（保序）
      "cycle": 0,                   // 控制器 cycle_n
      "milestone_index": null,      // 屬哪個 milestone；PLAN 級為 null
      "phase": "PLAN",              // PLAN | FIND | EXECUTE | CONTINUATION | SUBMIT
      "agent": "rough_planner",     // 角色 + 子步標籤（finder 會是 :community_select 等）
      "attempt": 1,                 // 該角色第幾次嘗試（重試 >1）
      "turn": null,                 // executor 多輪時的輪號（execute_python=1, submit_final=2）

      "input": {                    // ＝ observation/context（模型那一刻逐字看到的）
        "system_ref": "096c30ce",   // → system_prompts["096c30ce"]（不 inline）
        "context": [                // 變動輸入，逐字（含被串進來的前步觀察）
          { "role": "user", "text": "Task ID: 2a163ab_1\nCurrent datetime: 2023-05-18T21:26:01\nTask instruction:\nLike all the venmo transactions ..." }
        ],
        "tools": [],                // 可呼叫的 function declarations（executor 才有）
        "tool_config": null         // {mode: ANY|AUTO, allowed: [...]}（executor 才有）
      },

      "output": {                   // ＝ thought + code（決策）
        "raw": "{\"thoughts\": \"...\", \"tasks\": [...]}",   // 模型逐字輸出
        "rationale": "The task likes venmo social-feed transactions filtered by ...", // thought（有就填）
        "parsed": { "thoughts": "...", "tasks": [ { "task": "Read my roommates ...", "app": "phone" }, { "task": "...", "app": "venmo" } ] }
      },

      "result": null,               // ＝ observation（執行 step 才有；見下）
      "metrics": { "prompt_tokens": 0, "completion_tokens": 0, "thoughts_tokens": 0, "wall_ms": 0, "retries": 0 }
    },

    // ── 範例：execute step（唯一會產生環境觀察的 step）──
    {
      "step_id": 5, "cycle": 1, "milestone_index": 0, "phase": "EXECUTE",
      "agent": "code_executor", "attempt": 1, "turn": 1,
      "input": {
        "system_ref": "54daf68f",
        "context": [ { "role": "user", "text": "Task: ...\nMilestone [] (1/2): Intent: Read my roommates ...\nCode plan ...\nAllowed API specs (authoritative JSON): [ { \"app_name\": \"phone\", \"api_name\": \"search_contacts\", \"parameters\": [...], \"response_schemas\": {...} } ]" } ],
        "tools": [ "execute_python", "submit_final" ],
        "tool_config": { "mode": "ANY", "allowed": ["execute_python", "submit_final"] }
      },
      "output": {
        "raw": { "function_call": { "name": "execute_python", "args": { "code": "import json\nroommates = []\n..." } } },
        "rationale": null,
        "parsed": { "action": "execute_python", "code": "import json\nroommates = []\nwhile True:\n    result = apis.phone.search_contacts(relationship=\"roommate\", ...)\n    ..." }
      },
      "result": {                   // ＝ observation_m：環境回傳
        "stdout": { "value": [ { "contact_id": 1660, "email": "an-harrison@gmail.com", ... } ], "summary": "Read my roommates ...", "answer": "null" },
        "api_trace": [
          { "call": "phone.search_contacts(relationship='roommate', page_index=0, page_limit=20)", "returns": "list[3]", "schema": { "contact_id": "int", "email": "str", "relationships": "list[1]" } }
        ]
      },
      "metrics": { "prompt_tokens": 0, "completion_tokens": 0, "wall_ms": 0, "retries": 0 }
    }

    // ... 其餘 step：community_select / seed_filter / dependency_round_N / code_planner /
    //     code_executor(turn 2 = submit_final，result.committed_variable="roommates") /
    //     continuation(ADVANCE/SUBMIT) / SUBMIT(completion_gate + evaluator) ...
  ]
}
```

## 3. 逐欄位說明（你要的「完整內容」）

### run_meta — 來源與可重現性
- `producer`：`oracle-claude`（我蒸餾用跑的）或 `gemini-live`（正常 baseline）。**論文 strong→weak 揭露靠這欄**。
- `model` / `agent_commit` / `adk_version` / `rpc_url`：釘住「哪個模型、哪版 code、哪版 ADK、連哪個環境」跑出來的。
- `created_at`：跑完後蓋時間（run 內不可取時間）。
- `impls`：PLAN/FIND/EXECUTE 各用哪個 impl（消融變體靠這欄分辨）。

### task — 問題本身
- `task_id` / `split` / `instruction` / `datetime` / `user_profile`：模型解題的完整外部條件。`user_profile` 是「當前使用者」情境（區分 me vs others）。

### config — 可重現性釘子（對齊論文 §3.2.2 disclosure）
- `sampling`：temp/seed/top_p/top_k/max_tokens。
- `catalog_version`：community / API 母體版本——**檢索消融跨變體必須同版本才可比**。
- `prompt_set_version`：prompt 模板 commit。

### result — 結果
- `evaluator`：真 `world.evaluate()` 的 passed/failed/total/cleared（**gold 是否 PASS 的唯一裁判**）。
- `final_answer` / `status`：提交內容與閘門結果。

### plan — 便利投影
- 分解出的 milestone 清單。等同 PLAN step 的 output，頂層放一份方便索引（viewer 的 milestone 導覽、蒸餾的骨架）。

### system_prompts — content-addressed 去重存放（完整但不肥）
- **以「實際 render 後的 system 文字」的 hash 為 key** 存放：`{hash: {agent, chars, text(全文)}}`。**不是假設 system 固定**，是 content-addressed：
  - 靜態 prompt（現況 code_planner/executor/continuation/rough_planner，finder 注入固定 behavior_guidelines）→ 收斂成 1 個 entry。
  - **動態 / 注入 skill 後的 prompt（未來 skill induction 落地後）→ 每個不同 render 各一個 entry**，全部逐字留、無損。
- step 裡只放 `input.system_ref = hash`（指向 render 後文字，不是模板）。
- ⚠️ **2026-06-05 親驗修正**：原寫「固定、存一次」太絕對。實況：4 個 ADK prompt 是 runtime 靜態（純常數 / import 期 f-string；`{` 多為 JSON 範例字面、非模板）；finder 走 `.format()`（`routing.py:733` 注 `behavior_guidelines`）動態組但注入固定檔 → 等效靜態；ADK 的 `{var}`-from-state 注入目前**沒用**（task 內容走 user message）。但 **skill 注入 system 後 system 就變每-call 動態**，故 dedup 必為 content-addressed、不可預設 fixed。
- 同時滿足：①逐字完整（論文 appendix 要 prompt 全文）②不肥（同一段 system 重複出現只存一次 hash；subagent_io.jsonl 的痛點就是每步重貼）。

### steps[] — 軌跡本體（**superset：吸收 subagent_io 全部 + 補 executor 逐字輸入**）
每個 step = 一個決策點 = 一組 (情境 → 決策 → 觀察)：
- **定位 + 狀態**：`step_id`(保序) / `cycle` / `milestone_index` / `phase` / `agent` / `attempt` / `turn` / **`status`**(SUCCEEDED|FAILED) / **`failure_code`** / **`warnings`**。
- **`input`（observation/context）**：
  - `system_ref`(render 後 system 的 hash) + `context`(逐字變動輸入，含串進來的前步觀察) + `tools` + `tool_config` + `injected_skills`(注了哪些 skill；OFF 恆 `[]`，skill 消融載體)。
  - **`subagent_input`**（結構化情境，吸收 subagent_io 的 `metadata`）：`prior_variables`(完整變數庫含 `value_json`/`type_name`/`source_milestone_id`) / `history`(continuation 歷史) / `milestones`(快照) / `planned_apps` / `prior_attempts_for_this_milestone` / `retry_reason` / `latest_continuation_rationale` / `last_framework_override`。
  - ⚠️ **executor 的 `context` 正是 subagent_io 缺的逐字輸入** → oracle run 由 shim io/ patch、gemini run 由 llm_io callback patch。
- **`output`（thought + code + 該角色完整產物）**：
  - 通用：`raw`(逐字) + `rationale` + `parsed`(結構化決策)。
  - **finder 專屬 `api_selection`**：`matched_apps` / `selected_communities` / `candidate_apis`(enriched 全 specs) / `routing_trace`(每步 planned/matched/invalid/fallback) / `candidate_count`。
  - **executor 專屬 `code_execute`**：`code` / `tool_calls` / `tool_call_count` / `parse_error` / `llm_raised` / **`repair_attempts`**(內部重試，每次含 code+stdout — contrastive 蒸餾金礦)；`executor_result`(committed 變數含 `value_json`) / `submission_candidate` / `finalize_called` / `milestone_done`。
- **`result`（observation_m）**：execute step 的環境回傳 — `stdout`(四欄 JSON) + `api_trace`(sandbox 實際 `apis.*` 呼叫 + 回傳 schema) + `committed_variables`。下一步會被串進它的 `input.context`。
- **`metrics`**：`wall_ms` / token 4 欄 / `llm_call_attempts` / `timeout_count` / finder 的 `dependency_rounds` / retry 5 欄(`retry_count`/`retry_backoff_ms`/`rate_limit_count`/`provider_error_count`/`last_error`)。
- **`_debug`（可選）**：`event_diagnostics`(raw ADK events) / `expected_output`。預設收進此子物件、可關，不污染主結構。

### Superset 對照（subagent_io.jsonl 每個欄位 → trajectory 落點）

| subagent_io.jsonl | → trajectory | 備註 |
|---|---|---|
| `phase` / `io_record.agent` / `status` / `output.failure_code` / `warnings` | `step.{phase,agent,status,failure_code,warnings}` | |
| `input.model_input{system_instruction,messages,model,generate_content_config}` | `step.input.system_ref`(system 去重) + `step.input.context` + `config.sampling` | plan/continuation |
| `input.model_calls[].model_input_raw{...}` | finder 每步展成獨立 step 的 `input` | finder（多通） |
| `input.subagent_input.metadata.{prior_variables,history,milestones,planned_apps,...}` | `step.input.subagent_input` | 結構化情境 |
| `output.raw_llm_text` / `parsed_plan` / `parsed_decision` / `parsed_json` | `step.output.{raw,parsed}` | |
| `output.parsed_api_selection.{matched_apps,selected_communities,candidate_apis,routing_trace,candidate_count}` | `step.output.api_selection` | finder 產物 |
| `output.parsed_json.code_execute.{code,tool_calls,repair_attempts,parse_error,...}` | `step.output.code_execute` | executor 產物 |
| `envelope.payload.executor_result.variables[]` / `submission_candidate` / `milestone_done` | `step.output.{executor_result,submission_candidate,...}` | |
| `output.model_calls[].{usage,retry}` / `metrics.{...,dependency_rounds,retry_*}` | `step.metrics` | |
| `io_record.expected_output` / `code_execute.event_diagnostics` | `step._debug` | debug，可關 |
| **（subagent_io 沒有）executor 逐字輸入 prompt** | `step.input.context`（由 shim/callback patch） | **trajectory 多出來的** |

## 4. 三個消費者怎麼讀（都用投影，不另存來源）

| 消費者 | 讀法 |
|---|---|
| **skill 蒸餾（induction）** | 投影成 MIND-Skill 風格 `τ = [(input.context, output.parsed, result)]` per step；可選擇丟掉 system_ref（固定鷹架）只留變動情境 + 決策 + 觀察。**這才是真 trajectory，不是動作清單。** |
| **view builder** | adapter `step → io_record`：`input`(含 system_ref 展開)→`rendered_prompt` / `model_input_raw`；`output`→envelope；`result.api_trace`→sandbox 視圖。**executor 那格不再空白**（因為 input 完整）。 |
| **論文 reproducibility** | `system_prompts`→appendix 全文；`config`→§3.2.2 釘子；`metrics`→token usage；整份 commit 進 repo。 |

## 5. 兩種 run 怎麼產出同一份 schema

- **oracle-claude**：shim 已逐 call 抓 `input`(逐字) + `output`(我的回應)；`result` 取自 executor 的 function_response(stdout) + sandbox_api_calls.jsonl(api_trace)；assembler join → trajectory。（`build_trajectory.py` 是種子，要擴成「保留逐字 input + result + system_prompt 去重」。）
- **gemini-live**：`before/after_model_callback` 逐 call 抓 `input`+`output`（[[oracle-shim-harness]] 旁的 llm_io spec 那條，但靶改成填這個 schema）；`result` 同上取 sandbox trace；run state/ledger 給 cycle/milestone/phase 標籤；同一支 assembler join。

→ **同一份 schema、同一支 assembler，前端 capture 兩種；oracle 與 baseline 完全可比。**

## 6. 待 user 拍板
1. schema 欄位接受嗎？（尤其 system_prompts 去重 + step 的 input/output/result 三段式）
2. 要不要我把 `build_trajectory.py` 擴成這版（先對 oracle run 產完整 trajectory），再把 gemini 的 assembler 接上？
3. viewer 走 adapter（不動 2649 行）還是之後改讀 trajectory？
