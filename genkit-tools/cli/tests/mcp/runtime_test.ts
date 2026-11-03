/**
 * Copyright 2026 Google LLC
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

import { RuntimeEvent } from '@genkit-ai/tools-common/manager';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { defineFlowTools } from '../../src/mcp/flows';
import { defineRuntimeTools } from '../../src/mcp/runtime';
import { defineTraceTools } from '../../src/mcp/trace';
import type { McpToolOptions } from '../../src/mcp/utils';

vi.mock('@genkit-ai/tools-common/utils', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@genkit-ai/tools-common/utils')>();
  return {
    ...actual,
    record: vi.fn(() => Promise.resolve()),
  };
});

describe('MCP runtime tools', () => {
  let registeredTools: Record<string, (opts: any) => Promise<any>>;
  let mockServer: any;
  let mockMcpManager: any;
  let options: McpToolOptions;

  beforeEach(() => {
    registeredTools = {};
    mockServer = {
      registerTool: vi.fn((name: string, _config: any, handler: any) => {
        registeredTools[name] = handler;
      }),
    };
    mockMcpManager = {
      getManagerWithDevProcess: vi.fn(),
      getManager: vi.fn(),
    };
    options = {
      projectRoot: '/workspace',
      explicitProjectRoot: false,
      timeout: 5000,
      manager: mockMcpManager,
    };
    defineRuntimeTools(mockServer, options);
    defineFlowTools(mockServer, options);
    defineTraceTools(mockServer, options);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('start_runtime formats standard Error messages', async () => {
    mockMcpManager.getManagerWithDevProcess.mockRejectedValue(
      new Error('app process exited with code 1')
    );

    const res = await registeredTools['start_runtime']({
      command: 'npm',
      args: ['run', 'dev'],
    });

    expect(res).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: 'Error: app process exited with code 1',
        },
      ],
    });
  });

  it('restart_runtime restarts the process and waits for the new runtime to register', async () => {
    let runtimeEventListener:
      | ((event: RuntimeEvent, rt: any) => void)
      | undefined;
    const processPromise = new Promise<void>(() => {});
    const mockRuntimeManager = {
      listRuntimes: vi.fn().mockReturnValue([{ id: 'old-runtime' }]),
      onRuntimeEvent: vi.fn((cb: any) => {
        runtimeEventListener = cb;
        return vi.fn();
      }),
      processManager: {
        restart: vi.fn(async () => {
          setTimeout(() => {
            runtimeEventListener?.(RuntimeEvent.ADD, { id: 'new-runtime' });
          }, 10);
        }),
        waitForProcess: vi.fn(() => processPromise),
      },
    };
    mockMcpManager.getManager.mockResolvedValue(mockRuntimeManager);

    const res = await registeredTools['restart_runtime']({});

    expect(mockRuntimeManager.processManager.restart).toHaveBeenCalledTimes(1);
    expect(mockRuntimeManager.onRuntimeEvent).toHaveBeenCalledTimes(1);
    expect(res).toEqual({
      content: [{ type: 'text', text: 'Done.' }],
    });
  });

  it('restart_runtime returns an error if the restarted process exits before registering', async () => {
    const mockRuntimeManager = {
      listRuntimes: vi.fn().mockReturnValue([{ id: 'old-runtime' }]),
      onRuntimeEvent: vi.fn(() => vi.fn()),
      processManager: {
        restart: vi.fn(async () => {}),
        waitForProcess: vi.fn(() =>
          Promise.reject(new Error('app process exited with code 1'))
        ),
      },
    };
    mockMcpManager.getManager.mockResolvedValue(mockRuntimeManager);

    const res = await registeredTools['restart_runtime']({});

    expect(res).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: 'Error: app process exited with code 1',
        },
      ],
    });
  });

  it('run_flow and get_trace format standard Error messages', async () => {
    mockMcpManager.getManager.mockRejectedValue(
      new Error('No runtime found at /workspace')
    );

    const flowRes = await registeredTools['run_flow']({
      flowName: 'myFlow',
    });
    expect(flowRes).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: 'Error: No runtime found at /workspace',
        },
      ],
    });

    const traceRes = await registeredTools['get_trace']({
      traceId: 'trace-123',
    });
    expect(traceRes).toEqual({
      isError: true,
      content: [
        {
          type: 'text',
          text: 'Error: No runtime found at /workspace',
        },
      ],
    });
  });
});
