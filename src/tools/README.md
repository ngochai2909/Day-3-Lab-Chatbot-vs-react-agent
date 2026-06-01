# Tools

This folder contains callable functions for the ReAct agent.

Each tool should have:

- `name`: The exact action name the LLM can call.
- `description`: A clear explanation of when and how to use the tool.
- `function`: The Python function that executes the action.

Example:

```python
{
    "name": "calculator",
    "description": "Run basic arithmetic. Input should be a math expression string.",
    "function": calculator,
}
```
