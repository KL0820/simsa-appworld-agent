from __future__ import annotations

from contextlib import aclosing

from google.adk.agents.invocation_context import InvocationContext
from pydantic import ValidationError

from adk_appworld_agent.contracts.subagent_input import SubagentInput
from adk_appworld_agent.contracts.subagent_output import (
    SubagentEnvelope,
    SubagentStatus,
)
from adk_appworld_agent.orchestration.cache.policy import CacheLayer
from adk_appworld_agent.orchestration.content_utils import (
    content_from_text,
    content_to_text,
)
from adk_appworld_agent.orchestration.subagent_output_store import SubagentOutputStore
from adk_appworld_agent.subagents.base import Subagent


class SubagentRunner:
    def __init__(
        self,
        *,
        subagent_output_store: SubagentOutputStore | None = None,
        cache_layer: CacheLayer | None = None,
        invocation_source: str = "live",
    ) -> None:
        self.subagent_output_store = subagent_output_store or SubagentOutputStore()
        self.cache_layer = cache_layer
        self.invocation_source = invocation_source

    async def invoke(
        self,
        ctx: InvocationContext,
        subagent: Subagent,
        subagent_input: SubagentInput,
    ) -> dict[str, dict[str, list[dict]]]:
        spec = subagent.cache_spec

        if spec is not None and self.cache_layer is not None:
            hit = self.cache_layer.lookup(spec, subagent_input)
            if hit is not None:
                replayed = self.cache_layer.adapt_replay(spec, hit, subagent_input)
                return self.subagent_output_store.append_delta(
                    ctx.session.state, replayed
                )

        sandboxed_ctx = ctx.model_copy(
            deep=True,
            update={
                "user_content": content_from_text(
                    subagent_input.model_dump_json(), role="user"
                )
            },
        )

        last_text: str = ""
        agent = subagent.build_agent()
        async with aclosing(agent.run_async(sandboxed_ctx)) as agen:
            async for event in agen:
                last_text = content_to_text(event.content)

        if not last_text:
            envelope = self._synthetic_failure(
                subagent=subagent,
                subagent_input=subagent_input,
                failure_code="SUBAGENT_OUTPUT_MISSING",
            )
            return self.subagent_output_store.append_delta(ctx.session.state, envelope)

        try:
            envelope = SubagentEnvelope.model_validate_json(last_text)
        except ValidationError:
            envelope = self._synthetic_failure(
                subagent=subagent,
                subagent_input=subagent_input,
                failure_code="SUBAGENT_OUTPUT_INVALID",
            )
            return self.subagent_output_store.append_delta(ctx.session.state, envelope)

        if (
            envelope.phase != subagent_input.phase
            or envelope.subagent_name != subagent.name
            or envelope.attempt != subagent_input.attempt
        ):
            envelope = self._synthetic_failure(
                subagent=subagent,
                subagent_input=subagent_input,
                failure_code="SUBAGENT_OUTPUT_INVALID",
            )
            return self.subagent_output_store.append_delta(ctx.session.state, envelope)

        if spec is not None and self.cache_layer is not None:
            self.cache_layer.save(
                spec, subagent_input, envelope, source=self.invocation_source
            )

        return self.subagent_output_store.append_delta(ctx.session.state, envelope)

    @staticmethod
    def _synthetic_failure(
        *,
        subagent: Subagent,
        subagent_input: SubagentInput,
        failure_code: str,
    ) -> SubagentEnvelope:
        return SubagentEnvelope(
            phase=subagent_input.phase,
            subagent_name=subagent.name,
            attempt=subagent_input.attempt,
            status=SubagentStatus.FAILED,
            payload={},
            failure_code=failure_code,
        )
