/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { logger } from '@genkit-ai/tools-common/utils';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { defineDocsTool } from '../mcp/docs';
import { defineFlowTools } from './flows';
import { defineInitPrompt } from './prompts/init';
import { defineRuntimeTools } from './runtime';
import { defineTraceTools } from './trace';
import { McpRuntimeManager, McpToolOptions } from './utils';

export async function startMcpServer(params: {
  projectRoot: string;
  explicitProjectRoot: boolean;
  timeout?: number;
  /** `--experimental-auth`. */
  auth?: boolean;
}) {
  const { projectRoot, explicitProjectRoot, timeout, auth } = params;
  logger.info(`Starting MCP server in: ${projectRoot}`);

  const server = new McpServer(
    {
      name: 'Genkit MCP',
      version: '0.0.2',
      title: 'Genkit MCP Server',
      description:
        'Provides Genkit documentation lookup, runtime process management, flow discovery/execution, and execution trace inspection.',
    },
    {
      instructions: [
        'Use the Genkit MCP server whenever authoring, running, or debugging a Genkit project:',
        '1. Documentation (`list_genkit_docs`, `search_genkit_docs`, `read_genkit_docs`): Search and read official Genkit documentation before implementing or modifying Genkit features.',
        '2. Running & Debugging Flows (`start_runtime`, `list_flows`, `run_flow`, `get_trace`, `restart_runtime`, `kill_runtime`): Prefer these MCP tools over ad-hoc shell scripts when executing or troubleshooting Genkit flows and tools:',
        '   - Call `start_runtime` first with the app entry point (e.g. `{ command: "npx", args: ["tsx", "src/index.ts"] }`, `{ command: "uv", args: ["run", "python", "main.py"] }`, `{ command: "go", args: ["run", "."] }`, or `{ command: "dart", args: ["run", "bin/main.dart"] }`).',
        '   - Call `list_flows` to inspect registered flows and their input schemas, then call `run_flow` to execute a flow and obtain its `result` and `telemetry.traceId`.',
        '   - Call `get_trace` with that `traceId` to inspect the full execution trace, including every tool call input/output, model request/response, and span error.',
        '   - After editing code, call `restart_runtime` so changes take effect before re-running `run_flow`, and call `kill_runtime` when finished.',
      ].join('\n'),
    }
  );

  await defineDocsTool(server, projectRoot);
  defineInitPrompt(server);

  const manager = new McpRuntimeManager(auth);
  const options: McpToolOptions = {
    projectRoot,
    explicitProjectRoot,
    timeout,
    manager,
  };

  defineFlowTools(server, options);
  defineTraceTools(server, options);
  defineRuntimeTools(server, options);

  return new Promise(async (resolve) => {
    const transport = new StdioServerTransport();
    const cleanup = async () => {
      try {
        await manager.kill();
      } catch (e) {
        // ignore
      }
      resolve(undefined);
      process.exit(0);
    };
    transport.onclose = async () => {
      try {
        await manager.kill();
      } catch (e) {
        // ignore
      }
      resolve(undefined);
      process.exit(0);
    };
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    await server.connect(transport);
    logger.info('Genkit MCP Server running on stdio');
  });
}
