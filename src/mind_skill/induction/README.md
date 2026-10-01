# Component-specific skill induction

This package implements the induction stage of the offline learning loop.
For the complete workflow, see [the package overview](../README.md).

- `agent.py`: build and run the induction agent.
- `prompts.py`: component-specific initial induction instructions.
- `schemas.py`: validate candidates and render `SKILL.md`.
- `build_induction_input.py`: construct model input from a trajectory slice.
- `make_induction_slices.py`: separate planner, code-planner and executor examples.

The surrounding pipeline is organized by responsibility:

- `../trajectory/`: assemble and render source trajectories.
- `../deduction/`: reconstruct behavior using candidate skills.
- `../judges/`: outcome, reconstruction and rubric evaluation.
- `../textgrad/`: textual feedback and induction-prompt refinement.
- `../loop.py`: coordinate iterations and retain the best candidate.
- `../cli/`: training, reporting and artifact publication entry points.

From the repository root:

```bash
uv run python -m mind_skill.cli.run_induction_training --help
```

Training requires the separately retained source corpus and model credentials;
executor deduction also needs a dedicated AppWorld RPC sandbox. The portfolio
snapshot includes the implementation and final skills, not raw training data.
Source-trajectory provenance and model configuration for induction are separate
questions; a model choice in the training code does not establish how historical
input trajectories were collected.
