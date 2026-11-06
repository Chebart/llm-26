import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { appendFileSync, readFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { join } from "node:path";
import { openAICompletionsApi, type Model } from "@earendil-works/pi-ai/compat";
import { LangfuseSpanProcessor } from "@langfuse/otel";
import { setLangfuseTracerProvider, startObservation } from "@langfuse/tracing";
import { TracerProvider } from "@opentelemetry/sdk-trace";

type ToolSchema = {
  function: {
    name: string;
    description: string;
    parameters: Record<string, unknown>;
  };
};

type TurnContext = {
  trace_id?: string;
  parent_span_id?: string;
  turn_id?: string;
};

export default async function extension(pi: ExtensionAPI) {
  // Подключение к инструментам и идентификаторы текущего диалога.
  const baseUrl = process.env.LAB3_TOOL_URL!;
  const sessionId = process.env.LAB3_SESSION_ID!;
  const headers = {
    Authorization: `Bearer ${process.env.LAB3_TOOL_TOKEN ?? ""}`,
    "x-lab3-session": sessionId,
    "Content-Type": "application/json",
  };

  const turnContext = (): TurnContext =>
    JSON.parse(readFileSync(process.env.LAB3_TURN_CONTEXT!, "utf8"));

  // SDK связывает generations Node-процесса с родительским trace из Python.
  const telemetry = process.env.LANGFUSE_ENABLED?.toLowerCase() === "true"
    ? new TracerProvider({
        spanProcessors: [
          new LangfuseSpanProcessor({
            additionalHeaders: { "x-langfuse-ingestion-version": "3" },
          }),
        ],
      })
    : undefined;

  if (telemetry) setLangfuseTracerProvider(telemetry);

  const pendingGenerations = new Set<Promise<void>>();

  // Схемы общие для обоих агентов; MCP сообщает доступные через него операции.
  const response = await fetch(`${baseUrl}/schemas`, { headers });
  if (!response.ok) throw new Error(`Tool service: ${response.status}`);

  const schemas = await response.json() as ToolSchema[];
  const mcp = new Client({ name: "lab3-pi", version: "0.1.0" });

  await mcp.connect(
    new StreamableHTTPClientTransport(new URL(`${baseUrl}/mcp/`), {
      requestInit: { headers },
    }),
  );

  const available = await mcp.listTools();
  const mcpNames = new Set(available.tools.map(tool => tool.name));

  // Pi получает единый интерфейс инструментов, независимо от их транспорта.
  for (const { function: definition } of schemas) {
    pi.registerTool({
      name: definition.name,
      label: definition.name,
      description: definition.description,
      parameters: Type.Unsafe<Record<string, unknown>>(definition.parameters),

      async execute(toolCallId, params, signal) {
        let result: unknown;

        if (mcpNames.has(definition.name)) {
          const reply = await mcp.callTool(
            {
              name: definition.name,
              arguments: params as Record<string, unknown>,
            },
            undefined,
            { signal },
          );

          result = reply.structuredContent;

          if (result === undefined && Array.isArray(reply.content)) {
            const text = reply.content.find(block => block.type === "text");
            result = text && typeof text.text === "string"
              ? JSON.parse(text.text)
              : reply.content;
          }
        } else {
          const reply = await fetch(`${baseUrl}/tools/${definition.name}`, {
            method: "POST",
            headers,
            signal,
            body: JSON.stringify({ arguments: params, call_id: toolCallId }),
          });

          if (!reply.ok) throw new Error(`Tool ${definition.name}: HTTP ${reply.status}`);

          result = await reply.json();
        }

        return {
          content: [{ type: "text" as const, text: JSON.stringify(result) }],
          details: result,
        };
      },
    });
  }

  // Параметры Polza добавляются к штатному запросу Pi перед отправкой.
  const withModelSettings = (payload: unknown) => {
    return {
      ...(payload as Record<string, unknown>),
      temperature: Number(process.env.LAB3_TEMPERATURE ?? 0),
      reasoning: { effort: process.env.LAB3_REASONING_EFFORT ?? "none" },
      ...(process.env.LAB3_PROVIDER
        ? {
            provider: {
              only: [process.env.LAB3_PROVIDER],
              allow_fallbacks: false,
            },
          }
        : {}),
    };
  };

  const provider = JSON.parse(
    readFileSync(join(process.env.PI_CODING_AGENT_DIR!, "models.json"), "utf8"),
  ).providers.polza;

  let providerCalls = 0;
  let providerTurn: string | undefined;

  // Provider hook охватывает и основные запросы, и штатную суммаризацию Pi.
  pi.registerProvider("polza", {
    baseUrl: provider.baseUrl,
    api: "openai-completions",
    apiKey: process.env.POLZA_API_KEY,
    models: provider.models.map((model: Record<string, unknown>) => ({
      ...model,
      compat: provider.compat,
    })),

    streamSimple(model, context, options) {
      const current = turnContext();
      const turn = current.turn_id;

      // Счётчик ограничивает все запросы модели внутри одной реплики.
      if (turn !== providerTurn) {
        providerTurn = turn;
        providerCalls = 0;
      }

      providerCalls += 1;

      if (providerCalls > Number(process.env.LAB3_MAX_LLM_CALLS ?? 24)) {
        throw new Error("Maximum LLM calls reached (including compaction)");
      }

      const requestId = randomUUID();
      const generation = telemetry
        ? startObservation(
            "pi.llm",
            {
              model: model.id,
              input: context.messages,
              modelParameters: {
                temperature: Number(process.env.LAB3_TEMPERATURE ?? 0),
                max_tokens: options?.maxTokens ?? model.maxTokens,
              },
              metadata: {
                session_id: sessionId,
                turn_id: turn,
                provider: process.env.LAB3_PROVIDER,
                run_kind: process.env.LAB3_RUN_KIND,
                reasoning_effort: process.env.LAB3_REASONING_EFFORT,
                currency: "RUB",
              },
            },
            {
              asType: "generation",
              parentSpanContext: current.trace_id && current.parent_span_id
                ? {
                    traceId: current.trace_id,
                    spanId: current.parent_span_id,
                    traceFlags: 1,
                  }
                : undefined,
            },
          )
        : undefined;

      let gatewayCost: number | undefined;
      let stream;

      // Используем штатный stream Pi и сохраняем фактические payload и usage.
      try {
        stream = openAICompletionsApi().streamSimple(
          model as Model<"openai-completions">,
          context,
          {
            ...options,

            async onProviderStreamEvent(event, requestModel) {
              await options?.onProviderStreamEvent?.(event, requestModel);

              const usage = (event as { usage?: { cost_rub?: number } })?.usage;

              if (typeof usage?.cost_rub === "number") {
                gatewayCost = usage.cost_rub;

                appendFileSync(
                  join(process.env.PI_CODING_AGENT_DIR!, "billing.jsonl"),
                  JSON.stringify({
                    request_id: requestId,
                    turn_id: turn,
                    cost_rub: usage.cost_rub,
                  }) + "\n",
                );
              }
            },

            onPayload: async (payload, requestModel) => {
              const changed = await options?.onPayload?.(payload, requestModel);
              const prepared = withModelSettings(changed ?? payload);
              const parameters = prepared as {
                max_tokens?: number;
                max_completion_tokens?: number;
              };

              generation?.update({
                input: prepared,
                modelParameters: {
                  temperature: Number(process.env.LAB3_TEMPERATURE ?? 0),
                  max_tokens: parameters.max_tokens
                    ?? parameters.max_completion_tokens
                    ?? model.maxTokens,
                },
              });

              return prepared;
            },
          },
        );
      } catch (error) {
        generation?.update({ level: "ERROR", statusMessage: String(error) });
        generation?.end();
        throw error;
      }

      // result() не потребляет события: штатный цикл Pi читает исходный stream.
      const recorded = stream.result().then(message => {
        const failed = message.stopReason === "error" || message.stopReason === "aborted";

        generation?.update({
          output: message,
          level: failed ? "ERROR" : "DEFAULT",
          statusMessage: message.errorMessage,
          usageDetails: failed
            ? undefined
            : {
                input: message.usage.input,
                output: message.usage.output,
                cache_read: message.usage.cacheRead,
                cache_write: message.usage.cacheWrite,
              },
          metadata: {
            cost_rub: gatewayCost ?? (failed ? null : message.usage.cost.total),
            cost_source: gatewayCost === undefined ? "catalog_estimate" : "gateway",
            usage_partial: failed,
          },
        });

        generation?.end();
      }).catch(error => {
        generation?.update({ level: "ERROR", statusMessage: String(error) });
        generation?.end();
      });

      pendingGenerations.add(recorded);
      void recorded.finally(() => pendingGenerations.delete(recorded));

      return stream;
    },
  });

  // Ограничение итераций отделено от общего числа запросов, включая compaction.
  let steps = 0;

  pi.on("agent_start", () => {
    steps = 0;
  });

  pi.on("turn_start", () => {
    steps += 1;

    if (steps > Number(process.env.LAB3_MAX_STEPS ?? 12)) {
      throw new Error("Maximum agent iterations reached");
    }
  });

  // Перед завершением реплики и процесса дожидаемся отправки наблюдений.
  pi.on("agent_settled", async () => {
    await Promise.all(pendingGenerations);
    await telemetry?.forceFlush();
  });

  pi.on("session_shutdown", async () => {
    await Promise.all(pendingGenerations);
    await telemetry?.shutdown();
    await mcp.close();
  });
}
