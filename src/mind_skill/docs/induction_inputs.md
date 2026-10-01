# 每個 subagent 的 induction 輸入規格

把 MIND-Skill（2605.08670）的歸納法映射到本專案的多 subagent agent：**每個 subagent 各自蒸一份 skill**，餵進去的是「該 subagent 的軌跡切片」，不是整條 trajectory。

## 0. MIND-Skill 的做法（grounding）

- 餵 induction agent 的是 `task spec t + 成功軌跡 τ = {(thought_m, code_m, observation_m)}`，輸出一份 `SKILL.md`（Overview / When to Apply / Procedure / Key Patterns / Common Pitfalls）。
- **一次一條軌跡、一個 skill**（monolithic ReAct）；不是 90 條一起餵。
- 品質保證 = deduction agent 只憑 skill + task 在 live 環境**重建軌跡**，比對 + 過 evaluator（outcome loss）。

**多 subagent 對應**：MIND-Skill 一條軌跡只有一種 step；你的軌跡有 5 種角色的 step。所以 induction 變成 **per (subagent, task)**：對每個 subagent，把它那幾步抽出來、配上 task 指令，蒸一份對應該層的 skill。要泛化就跨「同 family 的 K 條軌跡」一起餵（仍只餵該 subagent 的切片）。

## 1. 共用的 induction 輸入信封

每次 induction 呼叫都長這樣（其餘一律不餵）：

```
TASK_INSTRUCTION:  <該題的指令>            # grounding，必帶
MILESTONE:         <該 step 的 milestone intent>  # PLAN 無、其餘必帶
TRAJECTORY_SLICE:  [ 該 subagent 的 (observation/context → decision → result) ]
                   # 跨 family 多題時，這裡放 K 份切片
→ 產出 SKILL.md（scoped 到該 subagent 的決策）
```

**一律不餵**：別的 subagent 的 step、`system_prompts` 全文（那是 agent 固定鷹架、不是情境）、`api_trace` 原始 rows（只餵摘要）、`prior_variables` 的 `value_json`（只餵 name+description）。

> ⚠️ MIND-Skill 的 observation 一定要留 —— 對每個 subagent，「observation」是讓決策有意義的那個情境（見下表），不能砍。

## 2. 每個 subagent 餵什麼（核心表）

token 數據來自 `runs/2a163ab_1`（單題、單 subagent 的 induction 切片）。

| subagent / skill | 注入點 | MIND-Skill 對應 (thought / code / observation) | **餵什麼（include）** | trim / drop | 單題切片 ≈ | 價值層級 |
|---|---|---|---|---|---|---|
| **rough_planner**（PLAN skill） | rough_planner prompt | thought=`plan.thoughts`／code=milestones／obs=∅（僅 task 指令） | task 指令 + `plan.thoughts` + milestones `[{app,intent}]` | 下游全段、system prompt | **~0.5K** | 推測（分解錯誤非主要失敗源） |
| **finder**（FIND skill） | finder community_select / seed_filter prompt | obs=**提供的候選社群/API + milestone**／code=`selected`／thought=`routing_trace` | milestone + **候選集（社群/API 的 name+description）** + 選了哪些 + `routing_trace` 摘要 | enriched specs(params/schema)、prior_vars | **~1K/milestone** | 中（檢索是貢獻面，但 retrieval-miss=0% → 可救空間小） |
| **code_planner**（plan skill） | code_planner prompt | obs=milestone+候選 API+prior vars／code=`CodePlanOutput` | milestone + **候選 API 名 + 關鍵 params**（如 `search_contacts(query,relationship,page_*)`） + prior var **名稱+描述** + `CodePlanOutput` | response_schema 全文、prior var 值、api_trace | **~1.5K** | **高**（executor 層=主戰場） |
| **code_executor**（execute skill）★ | code_executor prompt（`MILESTONE_NOT_DONE` 介入面） | obs=milestone+code plan+API specs+prior vars／code=**實際 code**／observation=stdout+api_trace | milestone + code plan + **用到的 API 簽名（param 名 + 回傳 shape）** + prior var 名稱/描述 + **實際 code** + **result 摘要**（成功+關鍵 shape） | **api_trace 原始 rows（摘要化）**、system prompt、prior var 值 | **~2.5K** | **最高**（這就是 THE skill；對應 MIND-Skill 的核心 procedural pattern） |
| **continuation**（control skill） | continuation prompt | obs=**執行結果**(last execute 摘要+status)／code=`ContinuationDecision`(next_action+rationale) | milestone 狀態 + last execute 摘要/status + decision + rationale | full history、full prior_variables、raw api_trace | **~1K** | 推測（控制決策模式） |

> ★ **code_executor 是首要目標**。skill_induction_design 的結論：retrieval-miss=0%、失敗都在執行層，所以 executor 的 procedural skill（join-key 選擇、分頁窮盡、verbatim pass-through、action-vs-query、membership-on-stable-key）才是 MIND-Skill 核心對應物。其餘四層你要做可以做，但價值較推測性、建議當次要/ablation 對照。

## 3. 對照整條 trajectory（為什麼這樣切）

單題整條 = **~119K tokens**，其中 `result.api_trace` 佔 50%（~60K，venmo feed 1089 筆原始資料）、`subagent_input` 佔 21%（prior_vars/history）、真正的決策/code 只佔 6%（~7K）。

→ 上表每個切片 0.5–2.5K，是因為**砍掉 api_trace 原始 rows + 別人的 step + system prompt + prior 值**，只留「該 subagent 的 (情境→決策→結果摘要)」。蒸一個 executor skill 從 family 3 題 ≈ 3×2.5K ≈ **7.5K tokens**，不是 3×119K。

## 4. 品質保證（per-subagent deduction）

照 MIND-Skill：把蒸出的某層 skill **注入該 subagent**（`injected_skills` 記錄），重跑該 family 的 held-out 題，過 evaluator（outcome）+ 看該 subagent 的決策有沒有照 skill 改變（reconstruction）。注入點就是上表「注入點」欄——這同時是 skill 消融的量測面。

## 5. 落地順序建議

1. 先做 **code_executor skill**（價值最高、對應 MIND-Skill 核心）：寫 `trajectory → executor 切片` 投影器 → 蒸第一份 `SKILL.md` → 注入重跑驗證。
2. 行有餘力再做 code_planner / finder（次要）；PLAN / continuation 當推測性對照。

> 前提：assembler 的 `result.api_trace` 要先改存**摘要版**（schema/分布/抽樣，非原始 rows），否則 executor 切片的 result 仍會被 feed 撐大。
