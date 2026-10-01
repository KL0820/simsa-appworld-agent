# Third-party sources

- **Google Agent Development Kit / google-genai**: installed dependencies for
  agent execution and model access; not authored by this project.
- **AppWorld**: benchmark, API definitions and evaluation infrastructure. Its
  task databases and source trajectories are not redistributed here. Retrieval
  artifacts were derived from the benchmark API descriptions. A copy of the
  license in the local AppWorld source is retained at `licenses/AppWorld-LICENSE`.
- **MIND-Skill**: the induction/deduction and textual-feedback method is adapted
  to this multi-subagent runtime; the underlying method is not claimed as original.
- **CUGA**: the final-answer extraction prompt and schema were adapted from its
  AppWorld final-answer component. See the source attribution in
  `src/adk_appworld_agent/submission/final_answer_extractor.py`. The upstream
  license and bundled notices are retained at `licenses/CUGA-LICENSE`.

This file records known origins; it does not grant a new project-wide license
or establish complete clearance for every derived research artifact. Before
public release, the author should confirm redistribution of the derived API
artifacts and learned skill library and select the license for original code.
