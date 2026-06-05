---
name: smriti-best-practices
description: Guidelines and instructions on how to optimally use the SMRITI long-term memory system across sessions.
---

# SMRITI Memory Guidelines

You have access to **SMRITI**, a neuro-inspired long-term memory architecture. Use the `smriti_*` MCP tools to maintain persistent context about the user, their preferences, project constraints, and historical decisions.

## How to use SMRITI

1. **Proactive Recall:** Call `smriti_recall` with 2-3 keywords before responding to retrieve past context about the user or the project.
2. **Immediate Encoding:** Call `smriti_encode` immediately when the user states a preference, constraint, architectural decision, or problem resolution. Don't wait until the end of the session.
3. **Private vs Shared:** By default, memories are shared. If the user explicitly asks you to remember something privately, use `private=True` in `smriti_encode` (or `amp.encode`).
4. **Knowledge Gaps:** You can use `smriti_knowledge_gaps` to see what topics you don't know much about and proactively ask the user clarifying questions.
5. **Confidence Checks:** Use `smriti_how_well_do_i_know` to verify if you have sufficient context before executing complex tasks.

Use these tools proactively to give the user a seamless, personalized experience without them having to repeat themselves.
