/**
 * Cursor SDK agent for the TMRL experiment orchestrator.
 *
 * Modes:
 *   decide <context.json>  -- Should the current experiment continue or stop?
 *   propose <context.json> -- Propose the next experiment after one completes.
 *
 * Output: JSON on stdout with the agent's decision.
 *
 * Usage (called by tmrl/tools/orchestrator.py):
 *   npx tsx scripts/orchestrator-agent.ts decide experiments/_agent_context.json
 *   npx tsx scripts/orchestrator-agent.ts propose experiments/_agent_context.json
 */

import { Agent } from "@cursor/sdk";
import { readFileSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function buildDecidePrompt(context: Record<string, unknown>): string {
  const snapshot = context.snapshot as Record<string, unknown> | undefined;
  const target = context.target_finish_time_s ?? 36.0;
  const elapsed = context.elapsed_hours ?? 0;
  const maxHours = context.max_hours ?? 8;
  const expEntry = context.exp_entry as Record<string, unknown> | undefined;

  return `You are an ML experiment analyst for a TrackMania RL agent.

TARGET: The model must finish the track in ${target} seconds or less.
METRIC: eval/finish_time_test_s (lower is better; only logged when the agent actually finishes a lap).

CURRENT EXPERIMENT: ${expEntry?.exp_id ?? "unknown"}
HYPOTHESIS: ${expEntry?.hypothesis ?? "N/A"}
CONFIG OVERRIDES: ${JSON.stringify(expEntry?.config_overrides ?? {}, null, 2)}
ELAPSED: ${elapsed} hours (max: ${maxHours} hours)

CURRENT METRICS SNAPSHOT:
${JSON.stringify(snapshot, null, 2)}

INSTRUCTIONS:
1. Use the metrics snapshot and exp_entry above; optional local files experiments/registry.jsonl or experiments/analysis/<exp_id>.json if present (not in git).
2. Analyze the metrics snapshot above.
3. Decide: should this experiment CONTINUE training, or STOP early?

Signs to STOP:
- Loss last > 100 or NaN, or Q-values exploded (max_q > 200).
- No positive best_finish_time_s AND worker_finish_count == 0 after 2+ hours.
- Gradient pre-clip >> 5? grad_clip with no finish progress.
- Worker episodes show only "no_progress_timeout" terminations.

Signs to CONTINUE:
- best_finish_time_s > 0 (even far above target) ? do NOT claim "never finished."
- worker_finish_count > 0 or worker_best_finish_time_s present.
- IQN loss in 30?90 with stable Q (max_q ~15?50) is normal here ? not automatic divergence.
- Eval finish times improving or worker finish count increasing.

OUTPUT FORMAT (JSON only, no markdown):
{"action": "continue" | "stop", "reason": "brief explanation referencing specific metrics"}`;
}

function buildProposePrompt(context: Record<string, unknown>): string {
  const target = context.target_finish_time_s ?? 36.0;
  const registry = context.registry as Array<Record<string, unknown>> | undefined;

  return `You are an ML experiment designer for a TrackMania RL agent.

TARGET: The model must finish the track in ${target} seconds or less.

EXPERIMENT HISTORY:
${JSON.stringify(registry ?? [], null, 2)}

INSTRUCTIONS:
1. Use EXPERIMENT HISTORY in this prompt; optional local experiments/decisions.md or experiments/analysis/<exp_id>.json if present (gitignored).
2. Read experiments/search_space.yaml for tunable parameters and their ranges.
3. Do not load all of experiments/analysis/ into context ? prefer per-exp JSON for the parent run only.
4. Analyze what has worked and what hasn't across all experiments.
5. Propose the NEXT experiment to run.

RULES:
- Change only 1-2 parameters from the best-performing parent experiment.
- Provide a clear hypothesis explaining WHY this change should help reach ${target}s.
- Reference specific metrics from prior experiments as evidence.
- Use the experiment_manager CLI to register: run the shell command:
  python -m tmrl.tools.experiment_manager register --parent <PARENT_ID> --hypothesis "<WHY>" --overrides '<JSON>'

After running the register command, append your reasoning to experiments/decisions.md (local, gitignored).

OUTPUT FORMAT (JSON only, no markdown):
{"action": "proposed", "exp_id": "<new_id>", "parent": "<parent_id>", "hypothesis": "...", "overrides": {...}}

If you cannot determine a good next experiment, output:
{"action": "no_proposal", "reason": "..."}`;
}

async function main(): Promise<void> {
  const [mode, contextPath] = process.argv.slice(2);

  if (!mode || !contextPath) {
    console.error("Usage: orchestrator-agent.ts <decide|propose> <context.json>");
    process.exit(1);
  }

  const contextRaw = readFileSync(contextPath, "utf-8");
  const context = JSON.parse(contextRaw) as Record<string, unknown>;

  const prompt = mode === "decide"
    ? buildDecidePrompt(context)
    : buildProposePrompt(context);

  const apiKey = process.env.CURSOR_API_KEY;
  if (!apiKey) {
    console.error("CURSOR_API_KEY not set");
    // Output a fallback JSON so the orchestrator doesn't crash
    if (mode === "decide") {
      console.log(JSON.stringify({ action: "continue", reason: "No CURSOR_API_KEY, defaulting to continue" }));
    } else {
      console.log(JSON.stringify({ action: "no_proposal", reason: "No CURSOR_API_KEY" }));
    }
    process.exit(0);
  }

  try {
    const result = await Agent.prompt(prompt, {
      apiKey,
      model: { id: "composer-2" },
      local: { cwd: REPO_ROOT },
    });

    if (result.status === "finished" && result.result) {
      // Extract JSON from the agent's response
      const text = result.result;
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        // Validate it's parseable
        const parsed = JSON.parse(jsonMatch[0]);
        console.log(JSON.stringify(parsed));
      } else {
        console.log(JSON.stringify({
          action: mode === "decide" ? "continue" : "no_proposal",
          reason: `Agent response did not contain JSON: ${text.slice(0, 200)}`,
        }));
      }
    } else {
      console.log(JSON.stringify({
        action: mode === "decide" ? "continue" : "no_proposal",
        reason: `Agent finished with status: ${result.status}`,
      }));
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.error(`Agent error: ${msg}`);
    console.log(JSON.stringify({
      action: mode === "decide" ? "continue" : "no_proposal",
      reason: `Agent error: ${msg}`,
    }));
  }
}

main();
