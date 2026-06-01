import re
from typing import List, Dict, Any
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ReActAgent:
    """
    A ReAct-style Agent that follows the Thought-Action-Observation loop.

    The LLM only *writes text* describing which tool it wants to use.
    This class is responsible for actually:
      1. Parsing that text to find the Action.
      2. Executing the corresponding Python tool.
      3. Feeding the Observation back into the prompt.
      4. Repeating until the LLM produces a "Final Answer".
    """

    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]], max_steps: int = 5, verbose: bool = False):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.verbose = verbose
        self.history = []

    def _log_step(self, msg: str):
        if self.verbose:
            print(msg)

    def get_system_prompt(self) -> str:
        """
        Instructs the LLM to follow the ReAct format and lists available tools.
        """
        tool_descriptions = "\n".join(
            [f"- {t['name']}: {t['description']}" for t in self.tools]
        )
        tool_names = ", ".join([t["name"] for t in self.tools])

        return f"""You are a careful shopping assistant that MUST use tools for every fact and every calculation. You never rely on your own memory or mental math.

You have access to ONLY these tools:
{tool_descriptions}

To solve the task, use this exact format:

Thought: describe what you need to do next.
Action: tool_name(argument)
Observation: (the system will fill this in with the tool result)
... (repeat Thought/Action/Observation as many times as needed)
Thought: I now know the final answer.
Final Answer: the final response to the user.

STRICT RULES:
1. Use ONLY the tools listed above ({tool_names}). Never invent a tool.
2. NEVER do arithmetic in your head. EVERY calculation (multiply, add, discount, subtotal, total) MUST go through the calculator tool, even simple ones like 2 * 999.
3. To get any product price, you MUST call lookup_product_price. Never guess a price.
4. To get any coupon discount, you MUST call get_discount. Never assume the percentage.
5. To get any shipping fee, you MUST call calc_shipping. Never guess the shipping cost.
6. Output the Action as plain text like calc_shipping(hanoi). Do NOT use markdown, backticks, or JSON.
7. Provide exactly ONE Action per step, then STOP and wait for the Observation. Do NOT write your own Observation.
8. Only output 'Final Answer:' AFTER you have gathered every required value via tools and computed the result with the calculator.

EXAMPLE (follow this multi-step pattern):
Question: Buy 2 keyboards with coupon WINNER, ship to hcm. Total?
Thought: I need the keyboard price.
Action: lookup_product_price(keyboard)
Observation: keyboard: $80
Thought: Subtotal for 2 keyboards via calculator.
Action: calculator(80 * 2)
Observation: 160
Thought: I need the WINNER discount percentage.
Action: get_discount(WINNER)
Observation: WINNER: 10% off
Thought: Apply 10% off using the calculator.
Action: calculator(160 * 0.9)
Observation: 144
Thought: I need the shipping fee to hcm.
Action: calc_shipping(hcm)
Observation: Shipping to hcm: $7
Thought: Add shipping with the calculator.
Action: calculator(144 + 7)
Observation: 151
Thought: I now know the final answer.
Final Answer: The total is $151.
"""

    def run(self, user_input: str) -> str:
        """
        Runs the ReAct loop until a Final Answer is produced or max_steps is hit.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        system_prompt = self.get_system_prompt()
        # The "scratchpad" accumulates the Thought/Action/Observation history.
        scratchpad = f"Question: {user_input}\n"
        steps = 0

        while steps < self.max_steps:
            steps += 1

            # 1. Ask the LLM to generate the next Thought + Action.
            result = self.llm.generate(scratchpad, system_prompt=system_prompt)
            content = result["content"]

            # Telemetry: track tokens/latency for this LLM call.
            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=result.get("usage", {}),
                latency_ms=result.get("latency_ms", 0),
            )

            logger.log_event(
                "AGENT_STEP",
                {"step": steps, "llm_output": content},
            )

            # 2. Check for a Final Answer -> stop the loop.
            final = self._parse_final_answer(content)
            if final is not None:
                self._log_step(f"  [step {steps}] Final Answer -> {final}")
                logger.log_event("AGENT_END", {"steps": steps, "status": "final_answer"})
                return final

            # 3. Parse the Action from the LLM output.
            action = self._parse_action(content)

            if action is None:
                # The LLM produced no parseable action and no final answer.
                self._log_step(f"  [step {steps}] PARSER_ERROR (no Action/Final Answer found)")
                logger.log_event(
                    "AGENT_ERROR",
                    {"step": steps, "error_code": "PARSER_ERROR", "raw": content},
                )
                # Nudge the model back on track and try again.
                scratchpad += (
                    f"{content}\n"
                    "Observation: Could not parse an Action. "
                    "Respond with either 'Action: tool_name(arg)' or 'Final Answer:'.\n"
                )
                continue

            tool_name, tool_arg = action

            # 4. Execute the tool and capture the Observation.
            observation = self._execute_tool(tool_name, tool_arg)
            self._log_step(f"  [step {steps}] Action: {tool_name}({tool_arg}) -> {observation}")

            logger.log_event(
                "TOOL_CALL",
                {"step": steps, "tool": tool_name, "arg": tool_arg, "observation": observation},
            )

            # 5. Append everything to the scratchpad for the next iteration.
            scratchpad += f"{content}\nObservation: {observation}\n"

        # If we exit the loop without a Final Answer -> timeout.
        logger.log_event("AGENT_END", {"steps": steps, "status": "max_steps_reached"})
        return "I could not complete the task within the allowed number of steps."

    def run_iter(self, user_input: str):
        """
        Generator version of run(): yields one event dict per step so a UI can
        render the Thought-Action-Observation loop live.

        Yielded event shapes:
          {"type": "thought_action", "step", "raw", "thought"}
          {"type": "tool_call",      "step", "tool", "arg", "observation"}
          {"type": "parser_error",   "step", "raw"}
          {"type": "final",          "step", "answer"}
          {"type": "timeout",        "step"}
        Each event also carries "usage" and "latency_ms" for the LLM call.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        system_prompt = self.get_system_prompt()
        scratchpad = f"Question: {user_input}\n"
        steps = 0

        while steps < self.max_steps:
            steps += 1

            result = self.llm.generate(scratchpad, system_prompt=system_prompt)
            content = result["content"]
            usage = result.get("usage", {})
            latency_ms = result.get("latency_ms", 0)

            tracker.track_request(
                provider=result.get("provider", "unknown"),
                model=self.llm.model_name,
                usage=usage,
                latency_ms=latency_ms,
            )
            logger.log_event("AGENT_STEP", {"step": steps, "llm_output": content})

            # Final answer?
            final = self._parse_final_answer(content)
            if final is not None:
                logger.log_event("AGENT_END", {"steps": steps, "status": "final_answer"})
                yield {
                    "type": "final", "step": steps, "answer": final,
                    "usage": usage, "latency_ms": latency_ms,
                }
                return

            # Action?
            action = self._parse_action(content)
            if action is None:
                logger.log_event(
                    "AGENT_ERROR",
                    {"step": steps, "error_code": "PARSER_ERROR", "raw": content},
                )
                yield {
                    "type": "parser_error", "step": steps, "raw": content,
                    "usage": usage, "latency_ms": latency_ms,
                }
                scratchpad += (
                    f"{content}\n"
                    "Observation: Could not parse an Action. "
                    "Respond with either 'Action: tool_name(arg)' or 'Final Answer:'.\n"
                )
                continue

            tool_name, tool_arg = action
            thought = self._parse_thought(content)
            yield {
                "type": "thought_action", "step": steps, "raw": content,
                "thought": thought, "tool": tool_name, "arg": tool_arg,
                "usage": usage, "latency_ms": latency_ms,
            }

            observation = self._execute_tool(tool_name, tool_arg)
            logger.log_event(
                "TOOL_CALL",
                {"step": steps, "tool": tool_name, "arg": tool_arg, "observation": observation},
            )
            yield {
                "type": "tool_call", "step": steps, "tool": tool_name,
                "arg": tool_arg, "observation": observation,
            }

            scratchpad += f"{content}\nObservation: {observation}\n"

        logger.log_event("AGENT_END", {"steps": steps, "status": "max_steps_reached"})
        yield {"type": "timeout", "step": steps}

    def _parse_thought(self, text: str):
        """Extract the first 'Thought:' line, if any."""
        match = re.search(r"Thought:\s*(.+?)(?:\n|Action:|Final Answer:|$)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_final_answer(self, text: str):
        """Return the final answer string if present, else None."""
        match = re.search(r"Final Answer:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _parse_action(self, text: str):
        """
        Extract (tool_name, argument) from a line like:
            Action: calculator(2 * 999)
        Returns None if no action is found.
        """
        # Find the LAST action in the text (in case the model wrote several lines).
        matches = re.findall(r"Action:\s*([a-zA-Z_]\w*)\s*\((.*?)\)", text, re.DOTALL)
        if not matches:
            return None
        tool_name, tool_arg = matches[-1]
        return tool_name.strip(), tool_arg.strip()

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Look up the tool by name and call its Python function.
        """
        for tool in self.tools:
            if tool["name"] == tool_name:
                try:
                    return str(tool["function"](args))
                except Exception as exc:
                    return f"Error executing {tool_name}: {exc}"
        # Tool not found -> this is a hallucination by the LLM.
        available = ", ".join([t["name"] for t in self.tools])
        return f"Tool '{tool_name}' not found. Available tools: {available}."
