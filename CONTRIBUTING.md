# Contributing to AgentFlow

Thanks for your interest in contributing to AgentFlow. This project explores agent-generated LangGraphs (flows and loops, not linear prompt chains) and tooling for evaluating and improving agentic systems — AgentOps. Contributions are welcome across design, evaluation, tooling, docs, and integration work.

## Ways to contribute
- Open issues describing bugs, feature requests, or design questions.
- Propose and implement features via pull requests.
- Add tests, benchmarks, and reproducible evaluation runs.
- Improve docs, examples, and templates (example flows, YAML, viewer pages).
- Share ideas for metrics and evaluation suites for agentic systems.

## Help wanted / good first ideas
If you want somewhere to start, consider:
- Add reproducible example flows and evaluation harnesses.
- Implement metrics collection for runs (latency, success, hallucination rate, throughput).
- Create more adapters (LLM providers, environment connectors).
- Improve the viewer UI for visualizing LangGraphs and run traces.
- Build end-to-end tests and CI for live agent runs (sandboxed).
- Implement automated self-improvement experiments: collect run metrics and use them to refine LangGraph generation.
- Extend the LangGraph-driven CLI pipeline with new analysis or guardrail nodes.

## How to submit
1. Fork the repo and create a topic branch: `git checkout -b feat/your-feature`
2. Add tests where appropriate and run them locally.
3. Commit with clear messages and open a pull request against `main`.
4. Link the PR to any relevant issue and describe the motivation, approach, and testing.

## Coding style & tests
- Follow existing project conventions (Python, pyproject-managed).
- Add or update tests in `tests/` for any behavior changes.
- Run tests with: `pytest -q`

## Issue and PR labels
Suggested labels to use:
- help wanted
- good first issue
- enhancement
- bug
- docs
- discussion

Maintainers may add or update labels as needed.

## Discussion & contact
Prefer issues for design discussion and RFCs. For quick sync or proposals, open an issue with the "discussion" or "help wanted" label.

## Licensing & CI
By contributing, you agree to abide by the project's license (see repository root). Please ensure CI passes for PRs that add logic or examples.

---

AgentFlow vision: teaching AI to learn how to learn by letting agents autonomously create LangGraphs (flows, loops, and conditional logic). We can collect run metrics, evaluate agentic systems as a whole, and bootstrap domain-specific agentic systems via iterative improvement. Your contributions will help push AgentOps forward.

Repository: https://github.com/stancsz/AgentFlow
