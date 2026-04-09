import { Router, type IRouter, type Request, type Response } from "express";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { GoogleGenAI } from "@google/genai";
import { authMiddleware } from "../../middlewares/auth";
import { isModelEnabled } from "../../lib/modelGroups";

// Fetch a remote image URL and return a base64 data URI
async function fetchImageAsBase64(url: string): Promise<string> {
  const res = await fetch(url, {
    headers: {
      "User-Agent": "Mozilla/5.0 (compatible; AI-Proxy/1.0)",
      "Accept": "image/*,*/*;q=0.8",
    },
    redirect: "follow",
  });
  if (!res.ok) throw new Error(`Failed to fetch image: ${res.status} ${url}`);
  const contentType = (res.headers.get("content-type") ?? "image/jpeg").split(";")[0]!;
  const buf = await res.arrayBuffer();
  const b64 = Buffer.from(buf).toString("base64");
  return `data:${contentType};base64,${b64}`;
}

// Convert image_url parts: replace remote URLs with base64 data URIs
// so the Replit AI Integrations proxy doesn't need to fetch external URLs itself
async function resolveImageUrls(messages: OAIMessage[]): Promise<OAIMessage[]> {
  return Promise.all(
    messages.map(async (msg) => {
      if (!Array.isArray(msg.content)) return msg;
      const resolvedContent = await Promise.all(
        msg.content.map(async (part) => {
          if (
            part.type === "image_url" &&
            typeof (part as { image_url?: { url: string } }).image_url?.url === "string"
          ) {
            const { url } = (part as { type: "image_url"; image_url: { url: string } }).image_url;
            if (!url.startsWith("data:")) {
              try {
                const dataUri = await fetchImageAsBase64(url);
                return { ...part, image_url: { ...(part as { image_url: object }).image_url, url: dataUri } };
              } catch {
                // keep original if fetch fails
              }
            }
          }
          return part;
        })
      );
      return { ...msg, content: resolvedContent };
    })
  );
}

const router: IRouter = Router();

const openai = new OpenAI({
  apiKey: process.env["AI_INTEGRATIONS_OPENAI_API_KEY"] ?? "dummy",
  baseURL: process.env["AI_INTEGRATIONS_OPENAI_BASE_URL"],
});

const anthropic = new Anthropic({
  apiKey: process.env["AI_INTEGRATIONS_ANTHROPIC_API_KEY"] ?? "dummy",
  baseURL: process.env["AI_INTEGRATIONS_ANTHROPIC_BASE_URL"],
});

const gemini = new GoogleGenAI({
  apiKey: process.env["AI_INTEGRATIONS_GEMINI_API_KEY"] ?? "dummy",
  httpOptions: { baseUrl: process.env["AI_INTEGRATIONS_GEMINI_BASE_URL"] },
});

const openrouter = new OpenAI({
  apiKey: process.env["AI_INTEGRATIONS_OPENROUTER_API_KEY"] ?? "dummy",
  baseURL: process.env["AI_INTEGRATIONS_OPENROUTER_BASE_URL"],
});

// ----------------------------------------------------------------------
// Claude model config (max_tokens per spec - must not change)
// ----------------------------------------------------------------------
const CLAUDE_MAX_TOKENS: Record<string, number> = {
  "claude-haiku-4-5": 8096,
  "claude-sonnet-4-5": 64000,
  "claude-sonnet-4-6": 64000,
  "claude-opus-4-1": 64000,
  "claude-opus-4-5": 64000,
  "claude-opus-4-6": 64000,
};

function getClaudeMaxTokens(model: string): number {
  return CLAUDE_MAX_TOKENS[model] ?? 64000;
}

function stripClaudeSuffix(model: string): {
  baseModel: string;
  thinkingEnabled: boolean;
  thinkingVisible: boolean;
} {
  if (model.endsWith("-thinking-visible")) {
    return {
      baseModel: model.slice(0, -"-thinking-visible".length),
      thinkingEnabled: true,
      thinkingVisible: true,
    };
  }
  if (model.endsWith("-thinking")) {
    return {
      baseModel: model.slice(0, -"-thinking".length),
      thinkingEnabled: true,
      thinkingVisible: false,
    };
  }
  return { baseModel: model, thinkingEnabled: false, thinkingVisible: false };
}

// ----------------------------------------------------------------------
// Type aliases for incoming OpenAI-format request body
// ----------------------------------------------------------------------
type OAIContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: { url: string; detail?: string } }
  | { type: "tool_result"; tool_use_id?: string; content?: string } // not real OAI but keep safe
  | Record<string, unknown>;

interface OAIMessage {
  role: string;
  content: string | OAIContentPart[] | null;
  name?: string;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
  }>;
  tool_call_id?: string; // role === "tool"
}

interface OAITool {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;
  };
}

type OAIToolChoice =
  | "auto"
  | "none"
  | "required"
  | { type: "function"; function: { name: string } };

interface ChatBody {
  model: string;
  messages: OAIMessage[];
  stream?: boolean;
  // generation params
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  n?: number;
  stop?: string | string[];
  seed?: number;
  presence_penalty?: number;
  frequency_penalty?: number;
  logprobs?: boolean;
  top_logprobs?: number;
  user?: string;
  // tools
  tools?: OAITool[];
  tool_choice?: OAIToolChoice;
  parallel_tool_calls?: boolean;
  // response_format
  response_format?: { type: string };
}

// ----------------------------------------------------------------------
// Message conversion: OpenAI -> Anthropic
// ----------------------------------------------------------------------

function oaiContentToAnthropic(
  content: string | OAIContentPart[] | null
): Anthropic.ContentBlockParam[] {
  if (content === null || content === undefined) return [];
  if (typeof content === "string") {
    // Anthropic rejects empty text blocks
    return content.length > 0 ? [{ type: "text", text: content }] : [];
  }

  const blocks: Anthropic.ContentBlockParam[] = [];
  for (const part of content) {
    if (part.type === "text" && typeof (part as { text?: string }).text === "string") {
      const text = (part as { text: string }).text;
      if (text.length === 0) continue; // Anthropic rejects empty text blocks
      blocks.push({ type: "text", text });
    } else if (part.type === "image_url") {
      const { url } = (part as { type: "image_url"; image_url: { url: string } }).image_url;
      if (url.startsWith("data:")) {
        const commaIdx = url.indexOf(",");
        const meta = url.slice(5, commaIdx); // strip "data:"
        const data = url.slice(commaIdx + 1);
        const mediaType = meta.split(";")[0] as Anthropic.Base64ImageSource["media_type"];
        blocks.push({ type: "image", source: { type: "base64", media_type: mediaType, data } });
      } else {
        blocks.push({ type: "image", source: { type: "url", url } });
      }
    }
    // skip unknown part types
  }
  return blocks;
}

function convertMessagesToAnthropic(messages: OAIMessage[]): {
  system: string | undefined;
  messages: Anthropic.MessageParam[];
} {
  let system: string | undefined;
  const converted: Anthropic.MessageParam[] = [];

  // We may need to merge consecutive tool results into a single user message
  // (Anthropic requires user/assistant alternation)
  let pendingToolResults: Anthropic.ToolResultBlockParam[] = [];

  const flushToolResults = () => {
    if (pendingToolResults.length > 0) {
      converted.push({ role: "user", content: [...pendingToolResults] });
      pendingToolResults = [];
    }
  };

  for (const msg of messages) {
    // -- system ------------------------------------------------------
    if (msg.role === "system") {
      system = typeof msg.content === "string" ? msg.content
        : Array.isArray(msg.content)
          ? msg.content.filter(p => (p as { type: string }).type === "text").map(p => (p as { text: string }).text).join("\n")
          : "";
      continue;
    }

    // -- tool result (role === "tool") --------------------------------
    if (msg.role === "tool") {
      const resultContent =
        typeof msg.content === "string"
          ? msg.content
          : Array.isArray(msg.content)
            ? msg.content.map(p => (p as { text?: string }).text ?? "").join("")
            : "";
      pendingToolResults.push({
        type: "tool_result",
        tool_use_id: msg.tool_call_id ?? "",
        content: resultContent,
      });
      continue;
    }

    // If we had accumulated tool results and now see a non-tool role, flush
    flushToolResults();

    // -- assistant ----------------------------------------------------
    if (msg.role === "assistant") {
      const content: Anthropic.ContentBlockParam[] = [];

      // text content
      const textBlocks = oaiContentToAnthropic(msg.content);
      content.push(...textBlocks);

      // tool_calls -> tool_use blocks
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          let parsedInput: Record<string, unknown> = {};
          try {
            parsedInput = JSON.parse(tc.function.arguments || "{}");
          } catch {
            parsedInput = {};
          }
          content.push({
            type: "tool_use",
            id: tc.id,
            name: tc.function.name,
            input: parsedInput,
          });
        }
      }

      if (content.length === 0) content.push({ type: "text", text: "" });
      converted.push({ role: "assistant", content });
      continue;
    }

    // -- user ---------------------------------------------------------
    const userContent = oaiContentToAnthropic(msg.content);
    if (userContent.length === 0) continue;
    converted.push({ role: "user", content: userContent });
  }

  flushToolResults();
  return { system, messages: converted };
}

// ----------------------------------------------------------------------
// Tool conversion: OpenAI tools -> Anthropic tools
// ----------------------------------------------------------------------

function convertToolsToAnthropic(tools: OAITool[]): Anthropic.Tool[] {
  return tools.map((t) => ({
    name: t.function.name,
    description: t.function.description,
    input_schema: (t.function.parameters ?? { type: "object", properties: {} }) as Anthropic.Tool["input_schema"],
  }));
}

function convertToolChoiceToAnthropic(
  tc: OAIToolChoice | undefined
): Anthropic.ToolChoiceParam | undefined {
  if (!tc || tc === "none") return undefined;
  if (tc === "auto") return { type: "auto" };
  if (tc === "required") return { type: "any" };
  if (typeof tc === "object" && tc.type === "function") {
    return { type: "tool", name: tc.function.name };
  }
  return { type: "auto" };
}

// ----------------------------------------------------------------------
// Gemini model config helpers
// ----------------------------------------------------------------------

function stripGeminiSuffix(model: string): { baseModel: string; thinkingEnabled: boolean } {
  if (model.endsWith("-thinking-visible")) {
    return { baseModel: model.slice(0, -"-thinking-visible".length), thinkingEnabled: true };
  }
  if (model.endsWith("-thinking")) {
    return { baseModel: model.slice(0, -"-thinking".length), thinkingEnabled: true };
  }
  return { baseModel: model, thinkingEnabled: false };
}

// ----------------------------------------------------------------------
// Message conversion: OpenAI -> Gemini
// ----------------------------------------------------------------------

type GeminiPart = { text: string };
type GeminiContent = { role: "user" | "model"; parts: GeminiPart[] };

function oaiContentToGeminiParts(content: string | OAIContentPart[] | null): GeminiPart[] {
  if (!content) return [{ text: "" }];
  if (typeof content === "string") return [{ text: content }];
  const parts: GeminiPart[] = [];
  for (const part of content) {
    if (part.type === "text" && typeof (part as { text?: string }).text === "string") {
      parts.push({ text: (part as { text: string }).text });
    }
  }
  return parts.length > 0 ? parts : [{ text: "" }];
}

function convertMessagesToGemini(messages: OAIMessage[]): {
  systemInstruction: string | undefined;
  contents: GeminiContent[];
} {
  let systemInstruction: string | undefined;
  const contents: GeminiContent[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      const text = typeof msg.content === "string"
        ? msg.content
        : (Array.isArray(msg.content)
          ? (msg.content as OAIContentPart[])
              .filter(p => p.type === "text")
              .map(p => (p as { text: string }).text)
              .join("\n")
          : "");
      systemInstruction = systemInstruction ? `${systemInstruction}\n${text}` : text;
    } else {
      const role: "user" | "model" = msg.role === "assistant" ? "model" : "user";
      const parts = oaiContentToGeminiParts(msg.content);
      // Merge consecutive same-role messages into one to satisfy Gemini alternation rule
      const last = contents[contents.length - 1];
      if (last && last.role === role) {
        last.parts.push(...parts);
      } else {
        contents.push({ role, parts });
      }
    }
  }

  // Gemini requires the first turn to be user; inject a stub if needed
  if (contents.length === 0 || contents[0].role !== "user") {
    contents.unshift({ role: "user", parts: [{ text: "." }] });
  }

  return { systemInstruction, contents };
}

// ----------------------------------------------------------------------
// SSE chunk helpers
// ----------------------------------------------------------------------

function makeChunk(
  id: string,
  model: string,
  delta: Record<string, unknown>,
  finishReason?: string | null
) {
  return {
    id,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta, finish_reason: finishReason ?? null }],
  };
}

function sseWrite(res: Response, data: unknown) {
  res.write(`data: ${JSON.stringify(data)}\n\n`);
}

function setSseHeaders(res: Response) {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("X-Accel-Buffering", "no");
}

// ----------------------------------------------------------------------
// Anthropic - streaming
// ----------------------------------------------------------------------

async function handleClaudeStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const { model, temperature, top_p, stop, tools, tool_choice } = body;
  const messages = await resolveImageUrls(body.messages);
  const { baseModel, thinkingEnabled, thinkingVisible } = stripClaudeSuffix(model);
  const modelMax = getClaudeMaxTokens(baseModel);
  // Thinking mode: always use model max (thinking tokens + output tokens both count).
  // Non-thinking: respect caller's max_tokens to honour token budget from main node,
  // but default to model max when unspecified so output is never truncated.
  const maxTokens = thinkingEnabled
    ? modelMax
    : (body.max_tokens && body.max_tokens > 0 ? Math.min(body.max_tokens, modelMax) : modelMax);

  const { system, messages: anthropicMessages } = convertMessagesToAnthropic(messages);

  // When tool_choice is "none", suppress tools entirely (Anthropic has no "none" option)
  const anthropicTools = (tools && tools.length > 0 && tool_choice !== "none")
    ? convertToolsToAnthropic(tools)
    : undefined;
  const anthropicToolChoice = (tool_choice && tool_choice !== "none")
    ? convertToolChoiceToAnthropic(tool_choice)
    : undefined;

  // Anthropic does not allow temperature / top_p when thinking is enabled
  const params: Record<string, unknown> = {
    model: baseModel,
    max_tokens: maxTokens,
    messages: anthropicMessages,
    stream: true,
  };
  if (system) params["system"] = system;
  if (!thinkingEnabled) {
    if (temperature !== undefined) params["temperature"] = temperature;
    else if (top_p !== undefined) params["top_p"] = top_p;
    if (stop) params["stop_sequences"] = Array.isArray(stop) ? stop : [stop];
  }
  // budget_tokens must leave room for visible output.
  // Cap at 10 000 and at most 60 % of maxTokens so at least 40 % remains for output.
  if (thinkingEnabled) params["thinking"] = { type: "enabled", budget_tokens: Math.min(10000, Math.floor(maxTokens * 0.6)) };
  if (anthropicTools) params["tools"] = anthropicTools;
  if (anthropicToolChoice) params["tool_choice"] = anthropicToolChoice;

  setSseHeaders(res);
  res.write(": init\n\n"); // flush connection immediately -- prevents proxy timeout before first AI token

  const id = `chatcmpl-${Date.now()}`;
  let inThinking = false;
  let inputTokens = 0; // captured from message_start, used in final usage chunk
  // Map Anthropic content-block index -> OAI tool_calls index (0-based among tool_use blocks only)
  const blockIdxToToolIdx: Record<number, number> = {};
  let toolCallCount = 0;
  let sseBodyStarted = false; // true after first sseWrite (headers have been flushed to wire)

  const keepaliveInterval = setInterval(() => {
    sseBodyStarted = true;
    res.write(": keepalive\n\n");
  }, 5000);

  try {
    const stream = anthropic.messages.stream(params as Anthropic.MessageCreateParamsStreaming);

    for await (const event of stream) {
      if (event.type === "message_start") {
        // Capture input token count for the final usage report
        inputTokens = event.message.usage?.input_tokens ?? 0;
        sseBodyStarted = true;
        sseWrite(res, makeChunk(id, model, { role: "assistant", content: "" }));

      } else if (event.type === "content_block_start") {
        const block = event.content_block;
        const idx = event.index;

        if (block.type === "thinking") {
          inThinking = true;
          if (thinkingVisible) {
            sseWrite(res, makeChunk(id, model, { content: "<thinking>\n" }));
          }
        } else if (block.type === "text") {
          if (inThinking && thinkingVisible) {
            sseWrite(res, makeChunk(id, model, { content: "\n</thinking>\n\n" }));
          }
          inThinking = false;
        } else if (block.type === "tool_use") {
          // Assign a sequential OAI tool call index (0-based) independent of content block index
          const toolIdx = toolCallCount++;
          blockIdxToToolIdx[idx] = toolIdx;
          sseBodyStarted = true;
          sseWrite(res, makeChunk(id, model, {
            tool_calls: [{
              index: toolIdx,
              id: block.id,
              type: "function",
              function: { name: block.name, arguments: "" },
            }],
          }));
        }

      } else if (event.type === "content_block_delta") {
        const delta = event.delta;
        const idx = event.index;

        if (delta.type === "thinking_delta") {
          if (thinkingVisible) {
            sseWrite(res, makeChunk(id, model, { content: delta.thinking }));
          }
        } else if (delta.type === "text_delta") {
          sseWrite(res, makeChunk(id, model, { content: delta.text }));
        } else if (delta.type === "input_json_delta") {
          // Use the mapped OAI tool call index, not the Anthropic content block index
          const toolIdx = blockIdxToToolIdx[idx] ?? 0;
          sseWrite(res, makeChunk(id, model, {
            tool_calls: [{
              index: toolIdx,
              function: { arguments: delta.partial_json },
            }],
          }));
        }

      } else if (event.type === "message_delta") {
        const stopReason = event.delta.stop_reason;
        const finishReason =
          stopReason === "tool_use" ? "tool_calls"
          : stopReason === "end_turn" ? "stop"
          : (stopReason ?? "stop");
        // Build accurate usage: input_tokens from message_start + output_tokens from message_delta
        const outputTokens = event.usage?.output_tokens ?? 0;
        const usage = {
          prompt_tokens: inputTokens,
          completion_tokens: outputTokens,
          total_tokens: inputTokens + outputTokens,
        };
        sseWrite(res, { ...makeChunk(id, model, {}, finishReason), usage });
      }
    }

    res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    if (sseBodyStarted) {
      // SSE already started -- send an error event so the client knows, then end cleanly
      try {
        sseWrite(res, {
          error: {
            message: streamErr instanceof Error ? streamErr.message : "Stream error",
            type: "stream_error",
          },
        });
        res.write("data: [DONE]\n\n");
      } catch { /* ignore write errors during cleanup */ }
      // Do NOT re-throw: error was communicated via SSE; let finally handle res.end()
    } else {
      // No SSE data written yet -- re-throw so outer catch can send a proper JSON error.
      // Do NOT call res.end() here; let the outer catch's res.json() do it.
      clearInterval(keepaliveInterval);
      throw streamErr;
    }
  } finally {
    clearInterval(keepaliveInterval);
    // End the response only when SSE body was started (success path or mid-stream error).
    // When !sseBodyStarted we re-threw above and the outer catch handles res.end().
    if (sseBodyStarted && !res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// Anthropic - non-streaming
// ----------------------------------------------------------------------

async function handleClaudeNonStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const { model, temperature, top_p, stop, tools, tool_choice } = body;
  const messages = await resolveImageUrls(body.messages);
  const { baseModel, thinkingEnabled, thinkingVisible } = stripClaudeSuffix(model);
  const modelMax = getClaudeMaxTokens(baseModel);
  // Same logic as streaming: thinking -> model max; otherwise honour caller, default to model max.
  const maxTokens = thinkingEnabled
    ? modelMax
    : (body.max_tokens && body.max_tokens > 0 ? Math.min(body.max_tokens, modelMax) : modelMax);

  const { system, messages: anthropicMessages } = convertMessagesToAnthropic(messages);

  // When tool_choice is "none", suppress tools entirely (Anthropic has no "none" option)
  const anthropicTools = (tools && tools.length > 0 && tool_choice !== "none")
    ? convertToolsToAnthropic(tools)
    : undefined;
  const anthropicToolChoice = (tool_choice && tool_choice !== "none")
    ? convertToolChoiceToAnthropic(tool_choice)
    : undefined;

  const params: Record<string, unknown> = {
    model: baseModel,
    max_tokens: maxTokens,
    messages: anthropicMessages,
    stream: false,
  };
  if (system) params["system"] = system;
  if (!thinkingEnabled) {
    if (temperature !== undefined) params["temperature"] = temperature;
    else if (top_p !== undefined) params["top_p"] = top_p;
    if (stop) params["stop_sequences"] = Array.isArray(stop) ? stop : [stop];
  }
  if (thinkingEnabled) params["thinking"] = { type: "enabled", budget_tokens: Math.min(10000, Math.floor(maxTokens * 0.6)) };
  if (anthropicTools) params["tools"] = anthropicTools;
  if (anthropicToolChoice) params["tool_choice"] = anthropicToolChoice;

  const response = await anthropic.messages.create(params as Anthropic.MessageCreateParamsNonStreaming);

  // Collect blocks
  let thinkingText = "";
  let bodyText = "";
  const toolCallResults: Array<{ id: string; name: string; input: unknown }> = [];

  for (const block of response.content) {
    if (block.type === "thinking") {
      thinkingText += (block as { thinking?: string }).thinking ?? "";
    } else if (block.type === "text") {
      bodyText += block.text;
    } else if (block.type === "tool_use") {
      toolCallResults.push({ id: block.id, name: block.name, input: block.input });
    }
  }

  const stopReason = response.stop_reason;
  const finishReason =
    stopReason === "tool_use" ? "tool_calls"
    : stopReason === "end_turn" ? "stop"
    : (stopReason ?? "stop");

  // Compose message
  let fullContent: string | null = bodyText || null;
  if (thinkingText && thinkingVisible) {
    fullContent = `<thinking>${thinkingText}</thinking>\n\n${bodyText}`;
  }

  const id = `chatcmpl-${Date.now()}`;

  const assistantMessage: Record<string, unknown> = {
    role: "assistant",
    content: fullContent,
  };

  if (toolCallResults.length > 0) {
    assistantMessage["tool_calls"] = toolCallResults.map((tc) => ({
      id: tc.id,
      type: "function",
      function: {
        name: tc.name,
        arguments: JSON.stringify(tc.input),
      },
    }));
  }

  res.json({
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: assistantMessage,
      finish_reason: finishReason,
    }],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
      total_tokens: response.usage.input_tokens + response.usage.output_tokens,
    },
  });
}

// ----------------------------------------------------------------------
// Gemini - streaming
// ----------------------------------------------------------------------

async function handleGeminiStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const { model, max_tokens, temperature, top_p } = body;
  const { baseModel, thinkingEnabled } = stripGeminiSuffix(model);
  const { systemInstruction, contents } = convertMessagesToGemini(body.messages);

  setSseHeaders(res);
  res.write(": init\n\n");

  const id = `chatcmpl-${Date.now()}`;

  const keepaliveInterval = setInterval(() => {
    res.write(": keepalive\n\n");
  }, 5000);

  try {
    const config: Record<string, unknown> = {
      maxOutputTokens: max_tokens ?? 8192,
    };
    if (temperature !== undefined) config["temperature"] = temperature;
    if (top_p !== undefined) config["topP"] = top_p;
    if (systemInstruction) config["systemInstruction"] = systemInstruction;
    if (thinkingEnabled) config["thinkingConfig"] = { thinkingBudget: -1 };

    sseWrite(res, makeChunk(id, model, { role: "assistant", content: "" }));

    const stream = await gemini.models.generateContentStream({
      model: baseModel,
      contents,
      config: config as Parameters<typeof gemini.models.generateContentStream>[0]["config"],
    });

    let inputTokens = 0;
    let outputTokens = 0;

    for await (const chunk of stream) {
      const text = chunk.text;
      if (text) {
        sseWrite(res, makeChunk(id, model, { content: text }));
      }
      if (chunk.usageMetadata) {
        inputTokens = chunk.usageMetadata.promptTokenCount ?? inputTokens;
        outputTokens = chunk.usageMetadata.candidatesTokenCount ?? outputTokens;
      }
    }

    sseWrite(res, makeChunk(id, model, {}, "stop"));
    sseWrite(res, {
      id, object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000), model,
      choices: [],
      usage: { prompt_tokens: inputTokens, completion_tokens: outputTokens, total_tokens: inputTokens + outputTokens },
    });
    res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      res.write("data: [DONE]\n\n");
    } catch { /* ignore */ }
  } finally {
    clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// Gemini - non-streaming
// ----------------------------------------------------------------------

async function handleGeminiNonStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const { model, max_tokens, temperature, top_p } = body;
  const { baseModel, thinkingEnabled } = stripGeminiSuffix(model);
  const { systemInstruction, contents } = convertMessagesToGemini(body.messages);

  const config: Record<string, unknown> = {
    maxOutputTokens: max_tokens ?? 8192,
  };
  if (temperature !== undefined) config["temperature"] = temperature;
  if (top_p !== undefined) config["topP"] = top_p;
  if (systemInstruction) config["systemInstruction"] = systemInstruction;
  if (thinkingEnabled) config["thinkingConfig"] = { thinkingBudget: -1 };

  const response = await gemini.models.generateContent({
    model: baseModel,
    contents,
    config: config as Parameters<typeof gemini.models.generateContent>[0]["config"],
  });

  const text = response.text ?? "";
  const inputTokens = response.usageMetadata?.promptTokenCount ?? 0;
  const outputTokens = response.usageMetadata?.candidatesTokenCount ?? 0;
  const id = `chatcmpl-${Date.now()}`;

  res.json({
    id,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{
      index: 0,
      message: { role: "assistant", content: text },
      finish_reason: "stop",
    }],
    usage: {
      prompt_tokens: inputTokens,
      completion_tokens: outputTokens,
      total_tokens: inputTokens + outputTokens,
    },
  });
}

// ----------------------------------------------------------------------
// OpenRouter - streaming (OpenAI-compatible, uses openrouter client)
// ----------------------------------------------------------------------

async function handleOpenRouterStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const resolvedMessages = await resolveImageUrls(body.messages);

  const params: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
    model: body.model,
    messages: resolvedMessages as OpenAI.ChatCompletionMessageParam[],
    stream: true,
    stream_options: { include_usage: true },
  };

  if (body.temperature !== undefined) params.temperature = body.temperature;
  if (body.top_p !== undefined) params.top_p = body.top_p;
  if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
  if (body.stop !== undefined) params.stop = body.stop as string | string[];

  setSseHeaders(res);
  res.write(": init\n\n");

  const keepaliveInterval = setInterval(() => {
    res.write(": keepalive\n\n");
  }, 5000);

  try {
    const stream = await openrouter.chat.completions.create(params);
    for await (const chunk of stream) {
      sseWrite(res, chunk);
    }
    res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      res.write("data: [DONE]\n\n");
    } catch { /* ignore */ }
  } finally {
    clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// OpenRouter - non-streaming
// ----------------------------------------------------------------------

async function handleOpenRouterNonStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const resolvedMessages = await resolveImageUrls(body.messages);

  const params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
    model: body.model,
    messages: resolvedMessages as OpenAI.ChatCompletionMessageParam[],
    stream: false,
  };

  if (body.temperature !== undefined) params.temperature = body.temperature;
  if (body.top_p !== undefined) params.top_p = body.top_p;
  if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
  if (body.stop !== undefined) params.stop = body.stop as string | string[];
  if (body.seed !== undefined) params.seed = body.seed;

  const response = await openrouter.chat.completions.create(params);
  res.json(response);
}

// ----------------------------------------------------------------------
// OpenAI - streaming
// ----------------------------------------------------------------------

async function handleOpenAIStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  setSseHeaders(res);
  res.write(": init\n\n"); // flush connection immediately -- prevents proxy timeout before first AI token

  const resolvedMessages = await resolveImageUrls(body.messages);

  // Pass through all OpenAI-compatible params
  const params: OpenAI.Chat.ChatCompletionCreateParamsStreaming = {
    model: body.model,
    messages: resolvedMessages as OpenAI.ChatCompletionMessageParam[],
    stream: true,
    stream_options: { include_usage: true },
  };

  if (body.temperature !== undefined) params.temperature = body.temperature;
  if (body.top_p !== undefined) params.top_p = body.top_p;
  if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
  if (body.stop !== undefined) params.stop = body.stop as string | string[];
  if (body.seed !== undefined) params.seed = body.seed;
  if (body.presence_penalty !== undefined) params.presence_penalty = body.presence_penalty;
  if (body.frequency_penalty !== undefined) params.frequency_penalty = body.frequency_penalty;
  if (body.n !== undefined) params.n = body.n;
  if (body.user !== undefined) params.user = body.user;
  if (body.response_format !== undefined) params.response_format = body.response_format as OpenAI.ResponseFormatText;
  if (body.logprobs !== undefined) params.logprobs = body.logprobs;
  if (body.top_logprobs !== undefined) params.top_logprobs = body.top_logprobs;
  if (body.tools && body.tools.length > 0) {
    params.tools = body.tools as OpenAI.ChatCompletionTool[];
    if (body.tool_choice !== undefined && body.tool_choice !== "none") {
      params.tool_choice = body.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
    }
    if (body.parallel_tool_calls !== undefined) {
      params.parallel_tool_calls = body.parallel_tool_calls;
    }
  }

  const keepaliveInterval = setInterval(() => {
    res.write(": keepalive\n\n");
  }, 5000);

  try {
    const stream = await openai.chat.completions.create(params);
    for await (const chunk of stream) {
      sseWrite(res, chunk);
    }
    res.write("data: [DONE]\n\n");
  } catch (streamErr) {
    try {
      sseWrite(res, {
        error: {
          message: streamErr instanceof Error ? streamErr.message : "Stream error",
          type: "stream_error",
        },
      });
      res.write("data: [DONE]\n\n");
    } catch { /* ignore write errors during cleanup */ }
  } finally {
    clearInterval(keepaliveInterval);
    if (!res.writableEnded) res.end();
  }
}

// ----------------------------------------------------------------------
// OpenAI - non-streaming
// ----------------------------------------------------------------------

async function handleOpenAINonStream(
  _req: Request,
  res: Response,
  body: ChatBody
) {
  const resolvedMessages = await resolveImageUrls(body.messages);

  const params: OpenAI.Chat.ChatCompletionCreateParamsNonStreaming = {
    model: body.model,
    messages: resolvedMessages as OpenAI.ChatCompletionMessageParam[],
    stream: false,
  };

  if (body.temperature !== undefined) params.temperature = body.temperature;
  if (body.top_p !== undefined) params.top_p = body.top_p;
  if (body.max_tokens !== undefined) params.max_tokens = body.max_tokens;
  if (body.stop !== undefined) params.stop = body.stop as string | string[];
  if (body.seed !== undefined) params.seed = body.seed;
  if (body.presence_penalty !== undefined) params.presence_penalty = body.presence_penalty;
  if (body.frequency_penalty !== undefined) params.frequency_penalty = body.frequency_penalty;
  if (body.n !== undefined) params.n = body.n;
  if (body.user !== undefined) params.user = body.user;
  if (body.response_format !== undefined) params.response_format = body.response_format as OpenAI.ResponseFormatText;
  if (body.logprobs !== undefined) params.logprobs = body.logprobs;
  if (body.top_logprobs !== undefined) params.top_logprobs = body.top_logprobs;
  if (body.tools && body.tools.length > 0) {
    params.tools = body.tools as OpenAI.ChatCompletionTool[];
    if (body.tool_choice !== undefined && body.tool_choice !== "none") {
      params.tool_choice = body.tool_choice as OpenAI.ChatCompletionToolChoiceOption;
    }
    if (body.parallel_tool_calls !== undefined) {
      params.parallel_tool_calls = body.parallel_tool_calls;
    }
  }

  const response = await openai.chat.completions.create(params);
  res.json(response);
}

// ----------------------------------------------------------------------
// Route
// ----------------------------------------------------------------------

router.post("/chat/completions", authMiddleware, async (req: Request, res: Response) => {
  try {
    const body = req.body as ChatBody;
    const { model, messages, stream } = body;

    if (!model || !messages) {
      res.status(400).json({
        error: { message: "model and messages are required", type: "invalid_request_error" },
      });
      return;
    }

    if (!isModelEnabled(model)) {
      res.status(403).json({
        error: {
          message: `Model '${model}' is currently disabled by the proxy administrator.`,
          type: "invalid_request_error",
          code: "model_disabled",
        },
      });
      return;
    }

    const isClaude      = model.startsWith("claude-");
    const isGemini      = model.startsWith("gemini-");
    const isOpenRouter  = !isClaude && !isGemini && model.includes("/");

    if (isClaude) {
      if (stream) {
        await handleClaudeStream(req, res, body);
      } else {
        await handleClaudeNonStream(req, res, body);
      }
    } else if (isGemini) {
      if (stream) {
        await handleGeminiStream(req, res, body);
      } else {
        await handleGeminiNonStream(req, res, body);
      }
    } else if (isOpenRouter) {
      if (stream) {
        await handleOpenRouterStream(req, res, body);
      } else {
        await handleOpenRouterNonStream(req, res, body);
      }
    } else {
      if (stream) {
        await handleOpenAIStream(req, res, body);
      } else {
        await handleOpenAINonStream(req, res, body);
      }
    }
  } catch (err: unknown) {
    req.log.error({ err }, "Error in /v1/chat/completions");
    if (!res.headersSent) {
      // Forward upstream API errors with their original status and message
      if (
        err &&
        typeof err === "object" &&
        "status" in err &&
        typeof (err as { status: unknown }).status === "number"
      ) {
        const apiErr = err as { status: number; message?: string; error?: { message?: string; type?: string } };
        const status = apiErr.status >= 400 ? apiErr.status : 502;
        const message = apiErr.error?.message ?? apiErr.message ?? "Upstream API error";
        const type = apiErr.error?.type ?? "upstream_error";
        res.status(status).json({ error: { message, type } });
      } else {
        res.status(500).json({ error: { message: "Internal server error", type: "server_error" } });
      }
    }
  }
});

export default router;
