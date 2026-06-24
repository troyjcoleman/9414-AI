# visual STRIPS planner

This folder runs a tiny local Python planner for the STRIPS planning demo.
The browser connection and replay formatting are handled by ai9414.
Your job is to return a grounded action plan for the current symbolic problem.

The same starter file works for:

- demo strips
- demo blocksworld

## install

Install the dependency with:

    pip install -r requirements.txt

## run

Start the local planner with:

    python solve_strips.py

## what to implement

Open solve_strips.py and look at:

- state_id(...)
- goal_satisfied(...)
- solve_planner(...)

Most of the domain-specific work is already done for you:

- get_initial_facts(problem) builds the initial symbolic state
- get_applicable_actions(problem, facts) returns currently legal grounded actions
- apply_action_signature(problem, facts, action) applies one legal action

Your code should not check whether the current problem is office delivery or
Blocks World. A correct BFS planner just searches over facts and applicable
actions.

For Blocks World, a fact looks like:

- ["on", "a", "c"]
- ["clear", "b"]
- ["handempty"]

For the Sussman anomaly, the initial stacks are bottom-to-top:

- c, then a
- b

The goal is:

- on(a, b)
- on(b, c)

The key lesson is that solving on(a, b) first can be unhelpful because b still
needs to move onto c. A complete BFS planner should find the interaction without
hard-coding that insight.

Your solver returns:

- algorithm
- status
- plan
- optional stats
