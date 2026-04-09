/**
 * Prompt-based tool calling fallback.
 *
 * Activated by setting `"x_use_prompt_tools": true` in the request body.
 * Useful for models that don't support native function calling.
 *
 * The proxy:
 *   1. Strips `tools` / `tool_choice` from the upstream request.
 *   2. Injects a system-prompt section that teaches the model the tool schema
 *      and instructs it to respond with a specific JSON format.
 *   3. After receiving the response, parses the JSON and converts it into
 *      OpenAI-compatible `tool_calls` or plain text.
 */

export interface PromptTool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

const INSTRUCTION = `
你是一个严格遵守输出格式的助手。根据用户的需求和下面提供的工具定义，判断是否需要调用工具。

### 工具定义
{{TOOLS_SCHEMA}}

### 输出格式（严格遵守，只输出 JSON，不要任何解释或 markdown）
- 如果不需要调用工具：{"type":"text","content":"你的回答"}
- 如果需要调用一个或多个工具：{"type":"tool_calls","calls":[{"name":"工具名","parameters":{"参数名":"值"}}]}

注意：只输出上述 JSON 格式中的一种，不能混用，不能在 JSON 外部添加任何文字。`.trim();

/**
 * Build the system-prompt injection for prompt-based tool calling.
 * Replaces `{{TOOLS_SCHEMA}}` with the compact JSON of the tools array.
 */
export function buildPromptToolsInstruction(tools: PromptTool[]): string {
  const schema = JSON.stringify(
    tools.map((t) => ({
      name: t.function.name,
      description: t.function.description ?? "",
      parameters: t.function.parameters ?? {},
    })),
    null,
    2,
  );
  return INSTRUCTION.replace("{{TOOLS_SCHEMA}}", schema);
}

export interface ParsedPromptToolsResult {
  isToolCall: boolean;
  calls?: Array<{ id: string; name: string; arguments: string }>;
  content: string;
}

/**
 * Parse the model's JSON response and convert to structured result.
 * Handles markdown code-fence stripping and loose JSON extraction.
 */
export function parsePromptToolsResponse(raw: string): ParsedPromptToolsResult {
  let text = raw.trim();

  // Strip markdown fences
  text = text.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim();

  // Find the outermost JSON object
  const start = text.indexOf("{");
  const end = text.lastIndexOf("}");
  if (start === -1 || end === -1) return { isToolCall: false, content: raw };

  try {
    const parsed = JSON.parse(text.slice(start, end + 1)) as Record<string, unknown>;

    if (parsed["type"] === "tool_calls" && Array.isArray(parsed["calls"])) {
      const calls = parsed["calls"] as Array<{ name: string; parameters?: Record<string, unknown> }>;
      return {
        isToolCall: true,
        calls: calls.map((c, i) => ({
          id: `call_pt_${Date.now()}_${i}`,
          name: String(c.name ?? ""),
          arguments: JSON.stringify(c.parameters ?? {}),
        })),
        content: "",
      };
    }

    if (parsed["type"] === "text" && typeof parsed["content"] === "string") {
      return { isToolCall: false, content: parsed["content"] };
    }
  } catch {
    // Invalid JSON — treat the whole response as plain text
  }

  return { isToolCall: false, content: raw };
}

/**
 * Build an OpenAI chat completion object from a prompt-tools result.
 * Used for the non-streaming path.
 */
export function buildCompletionFromPromptTools(
  result: ParsedPromptToolsResult,
  model: string,
  usage: { prompt_tokens: number; completion_tokens: number },
): Record<string, unknown> {
  const id = `chatcmpl-pt-${Date.now()}`;

  const message: Record<string, unknown> = {
    role: "assistant",
    content: result.isToolCall ? null : result.content,
  };

  if (result.isToolCall && result.calls) {
    message["tool_calls"] = result.calls.map((c) => ({
      id: c.id,
      type: "function",
      function: { name: c.name, arguments: c.arguments },
    }));
  }

  return {
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message,
        finish_reason: result.isToolCall ? "tool_calls" : "stop",
      },
    ],
    usage: {
      ...usage,
      total_tokens: usage.prompt_tokens + usage.completion_tokens,
    },
  };
}
