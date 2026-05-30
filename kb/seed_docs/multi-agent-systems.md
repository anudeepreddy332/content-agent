# Multi-Agent Systems — When and Why

## Definition
A multi-agent system (MAS) is a computational framework where multiple autonomous agents, each with its own perception, reasoning, and action capabilities, coexist and interact within a shared environment. Agents may have aligned, conflicting, or mixed goals, and they coordinate through communication protocols, negotiation, or emergent behaviors. In language-model–based MAS, each agent is typically an LLM-driven entity equipped with prompts, tools, and memory.

## Intuition
Complex problems can be decomposed into subtasks handled by specialized agents, akin to roles in a human team. Multiple agents can debate, verify each other’s outputs, or explore different solution paths in parallel, yielding more robust and creative outcomes than a single monolithic agent.

## Mathematical Formulation
A common model is the decentralized partially observable Markov decision process (Dec-POMDP), defined by a tuple (I, S, {A_i}, T, {O_i}, O, R), where I is the set of agents, S is the global state, A_i the action space of agent i, T(s, a, s') the transition function, O_i(s)→o_i the observation for agent i, and R(s, a) the shared reward. Each agent follows a policy π_i that maps its observation history to an action. The joint policy maximizes the expected cumulative reward: E[ Σ_t γ^t R(s_t, a_t) ]. In LLM-driven MAS, policies are not formally optimized but emergent from prompts and dialog, with coordination achieved via message passing of structured natural language.

## Estimation
No direct parameter estimation; system performance is evaluated experimentally using task completion rate, solution quality, communication cost, and robustness metrics. MAS behavior is validated on benchmark suites or simulators, often via repeated trials and inter-agent agreement measures.

## Assumptions
- Agents can send and receive messages over a reliable communication channel.
- Each agent has a bounded but sufficient reasoning capacity to interpret messages and adapt actions.
- Goals are either shared (cooperative), known and fixed (competitive), or can be inferred from interaction.
- The environment’s dynamics permit multi-agent intervention without catastrophic interference.

## Key Properties
- **Specialization**: Agents can be optimized for distinct subtasks.
- **Robustness**: Failure or drift of one agent can be compensated by others.
- **Emergent intelligence**: Collective behaviors like debate, voting, or role negotiation arise.
- **Scalability**: Workload can be distributed horizontally, though communication overhead grows.

## Failure Modes
- **Coordination failure**: Agents produce conflicting plans or deadlock on decisions.
- **Infinite chatter**: Prompt-driven loops without convergence.
- **Hallucinated agreements**: Agents pretend to agree without true coordination.
- **Cost blowup**: Multiple LLM calls per step compound inference expense.
- **Information cascade**: Early errors propagate when agents over-rely on peers.

## When to Use / Not Use
**Use**: When a task naturally decomposes into parallel or role-based subproblems (code generation + review, multi-perspective analysis, complex negotiation), when error correction via cross-checking is valuable, or when simulations of social dynamics are desired.

**Not use**: For simple single-step queries, latency-sensitive pipelines where serial agent chaining adds unacceptable delay, or when the added complexity of coordination outweighs accuracy gains.

## Variants / Extensions
- **Hierarchical teams**: Supervisor agent delegates to workers.
- **Competitive debate**: Agents argue opposing sides to improve answer quality.
- **Market-based coordination**: Resource allocation via bidding protocols.
- **Multi-agent reinforcement learning (MARL)**: Training policies with environment rewards.
- **CrewAI / AutoGen**: Frameworks for orchestrating conversational MAS.

## Minimal Example (Python)
```python
def agent1(observation):
    return f"Agent1 suggests: {observation.title()}"
def agent2(suggestion):
    if "?" in suggestion:
        return "Agent2: clarifying question needed"
    return "Agent2 agrees"
# Simple interaction
obs = "what is the weather?"
msg1 = agent1(obs)
msg2 = agent2(msg1)
final = msg2
```
## Claim-Dense Reference Facts

- AutoGen (Microsoft) uses ConversableAgent with two-agent or group-chat patterns; GroupChatManager handles speaker selection.
- CrewAI defines agents with role, goal, and backstory; tasks have description, expected_output, and agent assignment.
- LangGraph multi-agent uses Command(goto="agent_name") for handoffs between agents in a shared graph.
- Multi-agent debate improves factual accuracy by 11% on TruthfulQA compared to single-agent generation (Du et al., 2023).
- Communication cost scales as O(n²) messages for n fully-connected agents; hierarchical topologies reduce this to O(n).
- AutoGen's token usage grows approximately linearly with number of agents × average conversation turns.
- Agent specialization (code agent + critic agent) achieves higher HumanEval pass@1 than single generalist agent.
- LangGraph's `Send` API enables fan-out to multiple agents simultaneously for parallel subgraph execution.
- Swarm (OpenAI) implements agent handoffs via function returns; handoff functions return Agent objects directly.
- Production MAS systems require shared state management; LangGraph uses checkpointers (SqliteSaver, PostgresSaver) for persistence.