# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Google AI Interactions Antigravity model action."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from genkit_google_genai._interactions._client import create_interaction
from genkit_google_genai._interactions._converters import from_interaction_sync
from genkit_google_genai._interactions._options import ClientOptions
from genkit_google_genai._models._interactions_registry import antigravity_model_info
from genkit_google_genai._models._interactions_utils import (
    INPUT_HINT,
    MODEL_ID_HINT,
    TIMEOUT_DESCRIPTION,
    api_key_for_context,
    client_overrides_from_config,
    extract_version,
    lowercase_choice_list,
    partition_keys,
    pop_extra,
    reject_function_tools,
    reject_managed_extra,
    remove_client_option_overrides,
    steps_with_folded_system_instruction,
)
from genkit_google_genai._models._sdk_config import check_timeout_seconds, deep_merge, reject_bool_timeout
from genkit_google_genai._models._secrets import reject_request_config_api_key
from pydantic import Field, field_validator

from genkit import GenkitError, ModelResponse
from genkit.model import Model, ModelContext, ModelRequest
from genkit.plugin_api import GenkitConfig

DEFAULT_ENVIRONMENT: dict[str, str] = {'type': 'remote'}

CREATE_OPTION_KEYS = (
    'previous_interaction_id',
    'store',
    'environment',
    'response_modalities',
)

ANTIGRAVITY_EXTRA_DESCRIPTION = (
    "Interactions API fields this class doesn't declare, in the API's snake_case wire names "
    "(including `tools`, for example `[{'type': 'google_search'}, {'type': 'url_context'}]`). "
    'Merged into the top level of the create body after everything else, and nested objects merge key by key, '
    'so a key named here wins. Keys match ignoring case and underscores. Fields Genkit builds from the request '
    '(`agent` and `input`) and an empty `tools` list (`[]`) raise. A `None` value means unset and is skipped. '
    'Not validated, except that a top-level API key raises; do not put API keys here.'
)


class AntigravityConfig(GenkitConfig):
    """Antigravity model configuration.

    Unknown keys are rejected. An API field this class doesn't declare goes in
    ``extra``, which is deep-merged into the create body. Fields Genkit builds
    from the request (``agent``, ``input``) raise there.

    The agent enables ``google_search``, ``url_context``, and
    ``code_execution`` by default when ``tools`` is omitted. To restrict those
    defaults or pass Interactions server tools (such as ``mcp_server``), pass
    the exact allowlist of wire tool dicts in ``extra={'tools': [...]}``.
    ``extra={'tools': []}`` raises ``INVALID_ARGUMENT`` because the API reads
    an empty list as unset and turns all three defaults back on.

    ``timeout`` is the HTTP timeout for this request, in seconds.

    Example:
        ```python
        # 1. Default: omit `extra['tools']` to keep search, URL context, and sandbox bash on
        response = await ai.generate(
            model=GoogleAI.antigravity('antigravity-preview-09-2026'),
            prompt='Reconcile invoice subtotal $224.49 with 8.25% tax.',
        )
        # => create body omits `tools`

        # 2. Escape hatch: restrict the agent to web search + URL fetching (no sandbox bash)
        response = await ai.generate(
            model=GoogleAI.antigravity('antigravity-preview-09-2026'),
            prompt='Summarize the allergen policy at three Austin ramen shops.',
            config=AntigravityConfig(
                extra={'tools': [{'type': 'google_search'}, {'type': 'url_context'}]},
            ),
        )
        # => create body tools: [{'type': 'google_search'}, {'type': 'url_context'}]
        ```
    """

    base_url: str | None = None
    api_version: str | None = None
    timeout: float | None = Field(default=None, description=TIMEOUT_DESCRIPTION)
    custom_headers: dict[str, str] | None = None
    previous_interaction_id: str | None = None
    store: bool | None = None
    environment: str | dict[str, Any] | None = None
    response_modalities: list[Literal['text', 'image']] | None = None
    extra: dict[str, Any] | None = Field(default=None, description=ANTIGRAVITY_EXTRA_DESCRIPTION)

    @field_validator('timeout', mode='before')
    @classmethod
    def no_bool_timeout(cls, value: object) -> object:
        """``True`` is not a number of seconds."""
        return reject_bool_timeout(value)

    @field_validator('timeout')
    @classmethod
    def check_timeout(cls, value: float | None) -> float | None:
        """Reject 0, negative, inf and NaN; ``timeout`` is seconds."""
        return check_timeout_seconds(value)

    @field_validator('response_modalities', mode='before')
    @classmethod
    def fold_modalities_case(cls, value: object) -> object:
        """Accept TEXT/IMAGE the same as lowercase wire values."""
        return lowercase_choice_list(value)


# Fields Genkit builds from the request; extra can't replace them.
ANTIGRAVITY_MANAGED_EXTRA = {
    'agent': MODEL_ID_HINT,
    'input': INPUT_HINT,
}


def check_antigravity_extra_tools(name: str, extra: Mapping[str, Any]) -> None:
    """Reject empty or non-list ``extra['tools']`` before any HTTP call.

    The Interactions API reads ``tools: []`` as unset on the wire and turns all
    three built-in tools back on, so an empty list fails closed instead of
    looking like a sandbox lockdown.
    """
    for key, value in extra.items():
        if key.replace('_', '').lower() != 'tools':
            continue
        if not isinstance(value, list):
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=f'{name}: extra[{key!r}] must be a non-empty list of Interactions tool dicts.',
            )
        if not value:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f'{name}: extra[{key!r}] cannot be empty ([]). '
                    'The Interactions API reads an empty tools list as unset and '
                    'turns google_search, url_context, and code_execution back on; '
                    'pass at least one tool or omit tools to keep the defaults.'
                ),
            )


def create_antigravity_action(
    name: str,
    *,
    plugin_api_key: str | None,
    client_options: ClientOptions,
) -> Model:
    """Build a foreground model action for Antigravity."""
    version = extract_version(name)
    info = antigravity_model_info(version)

    async def run(request: ModelRequest[AntigravityConfig], ctx: ModelContext) -> ModelResponse:
        reject_request_config_api_key(request.config)
        config = request.config or AntigravityConfig()
        api_key = api_key_for_context(ctx.context, plugin_api_key)
        merged_options = client_options.merge(client_overrides_from_config(config))

        dumped = remove_client_option_overrides(config.model_dump(exclude_none=True))
        extra = pop_extra(dumped)
        reject_function_tools(
            name,
            request,
            message=(
                f'Function tools on {name} are not yet supported in this plugin. '
                "Built-in tools are on by default; pass server tool dicts in config extra={'tools': [...]}."
            ),
        )
        reject_managed_extra(name, extra, ANTIGRAVITY_MANAGED_EXTRA)
        check_antigravity_extra_tools(name, extra)
        # GenkitConfig rejects undeclared keys, so nothing is left over.
        create_options, _ = partition_keys(dumped, CREATE_OPTION_KEYS)
        create_kwargs: dict[str, Any] = {
            'agent': version,
            'input': steps_with_folded_system_instruction(request.messages),
            **create_options,
        }
        # Default missing environment to remote; the API rejects unsupported values.
        create_kwargs.setdefault('environment', DEFAULT_ENVIRONMENT)

        created = await create_interaction(api_key, deep_merge(create_kwargs, extra), merged_options)
        return from_interaction_sync(created)

    return Model(
        name,
        run,
        info=info,
        config_schema=AntigravityConfig,
    )
