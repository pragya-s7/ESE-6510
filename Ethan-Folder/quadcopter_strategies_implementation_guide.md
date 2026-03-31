# Quadcopter Strategies Implementation Guide

This document explains the current implementation of the three central functions in [quadcopter_strategies.py](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py):

- `get_rewards()`
- `get_observations()`
- `reset_idx()`

The goal of this guide is to explain what each function is doing, why it is doing it, and what design philosophy the current implementation is following.

## Big Picture
The strategy file is where we define the learning problem.

The physics simulator already handles:

- drone dynamics
- action application
- motor model
- collision sensing
- gate geometry
- episode stepping

Our strategy code decides:

- what counts as success
- what counts as failure or cheating
- what information the policy gets to see
- what initial states the policy is trained from

That means these three functions collectively define the policy's task, input, and training distribution.

## `get_rewards()`
Code reference: [quadcopter_strategies.py:68](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:68)

### Purpose
`get_rewards()` answers the question: what behavior do we want to encourage?

The current implementation is intentionally simpler than some earlier versions. It focuses on:

- forward progress toward the active gate
- valid gate traversal
- crash and cheating penalties

### Step 1: Read the drone state
At the beginning of the function, we read:

- drone position in world frame
- drone position in the current gate frame

This matters because:

- world-frame position is useful for measuring motion along the track
- gate-frame position is useful for deciding whether the drone actually crossed a gate correctly

### Step 2: Maintain previous-step state
The implementation lazily creates a few helper tensors:

- previous drone pose relative to the current gate
- previous drone world position
- last gate that was successfully passed
- previous pose relative to that last passed gate

These are not part of the original simulator state. We create them because gate traversal is fundamentally a temporal event. To know whether we crossed a gate, we need both the previous position and the current position.

### Step 3: Detect a valid gate crossing
A gate pass is not defined as "being near the gate." Instead, it is defined as a real geometric crossing.

The current logic requires all of the following:

1. The drone must move from the front side of the gate plane to the back side.
2. The crossing point must lie inside the gate opening.
3. The drone's motion must be aligned with the course direction from the current gate to the next gate.

Code reference: [quadcopter_strategies.py:99](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:99) through [quadcopter_strategies.py:127](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:127)

This is important because it prevents the policy from learning a cheap trick like oscillating near the gate plane or crossing in the wrong direction.

### Step 4: Detect cheating by re-crossing a passed gate
One of the main failure modes we saw was that the drone could pass a gate, then immediately cross back through it and exploit the waypoint logic.

To address that, we added explicit anti-cheat logic.

The implementation stores the index of the last gate that was legitimately passed. Then it checks whether the drone later crosses back through that same gate in reverse, still inside the gate opening.

If that happens, we mark it as cheating and treat it like a crash.

Code reference: [quadcopter_strategies.py:129](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:129) through [quadcopter_strategies.py:162](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:162)

Conceptually, this means:

- passing a gate is a one-way event
- you do not get to "farm" the same gate multiple times
- going back through a completed gate is considered invalid behavior

### Step 5: Compute a simple progress reward
Once valid gate logic is in place, the dense part of the reward is kept intentionally minimal.

We compute:

- current distance to the active target gate
- previous distance to that target gate
- positive forward improvement only

Then we clip it and add a small bonus when a gate is successfully passed.

Code reference: [quadcopter_strategies.py:164](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:164) through [quadcopter_strategies.py:181](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:181)

Why this design:

- it is simple
- it still gives the policy some shaping signal
- it is less hand-engineered than a large bundle of dense terms
- simpler reward design is often easier to debug and may reduce overfitting to simulator quirks

### Step 6: Crash handling
Finally, we read contact forces from the contact sensor.

A normal collision produces a crash signal. The anti-cheat signal is merged into the same crash pathway.

Code reference: [quadcopter_strategies.py:187](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:187)

This means both of these behaviors are treated similarly:

- physically crashing into something
- cheating by re-crossing a passed gate backwards

### Step 7: Form the final reward
The final reward currently has only two active components:

- `progress_goal`
- `crash`

Code reference: [quadcopter_strategies.py:194](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:194)

This is deliberate. The current reward philosophy is:

- keep the shaping simple
- avoid an overly dense reward
- encourage forward task completion
- punish invalid behavior strongly

## `get_observations()`
Code reference: [quadcopter_strategies.py:213](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:213)

### Purpose
`get_observations()` answers the question: what information should the policy see at each step?

The current observation design tries to be geometrically meaningful rather than just dumping raw world-state values.

### What the policy sees
The observation vector currently contains:

- drone linear velocity in body frame
- drone attitude as a 3x3 rotation matrix
- the 4 corners of the current gate in body frame
- the 4 corners of the next gate in body frame

Code reference: [quadcopter_strategies.py:268](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:268)

### Why gate corners?
Instead of giving only a gate center point, we provide the corners of the current and next gates.

This gives the policy richer geometric information:

- where the opening is
- how it is oriented
- how large it appears relative to the drone
- where the next gate sits after the current one

That helps the policy reason about both immediate traversal and short-horizon planning.

### Why body frame?
We transform gate geometry into the drone's body frame.

That is useful because the control problem is local:

- the drone needs to know where the gate is relative to itself
- body-frame features are often easier for a policy to use than absolute world coordinates

### Why use a rotation matrix?
The drone attitude is encoded as a full rotation matrix rather than Euler angles or quaternions.

Intuition:

- quaternions are compact but can be less intuitive to a network
- Euler angles can have discontinuities
- a rotation matrix is redundant, but it is smooth and explicit

That makes it a reasonable representation for learning.

### Extra bookkeeping inside observations
This function also updates:

- wrapped and unwrapped yaw tracking
- previous action buffer

Code reference: [quadcopter_strategies.py:257](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:257)

This bookkeeping supports tracking of motion over time and general episode state.

## `reset_idx()`
Code reference: [quadcopter_strategies.py:283](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:283)

### Purpose
`reset_idx()` answers the question: from what situations should the agent learn?

This function defines the training distribution of starting states.

That matters a lot. Even a good reward and observation design can struggle if the reset distribution is too narrow or unrealistic.

### Core training idea
During training, the drone does not always start at gate 0.

Instead:

- 10% of resets are forced to start near gate 0
- the remaining resets sample a random gate

Code reference: [quadcopter_strategies.py:339](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:339) through [quadcopter_strategies.py:367](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:367)

Why do this?

- it keeps exposure to the start of the track
- it also gives the policy experience from many local gate-approach situations
- this improves sample efficiency because the agent does not need to fly the whole lap just to practice later gates

### Spawn offsets
After sampling a waypoint, we place the drone behind that gate using local offsets.

For gate 0 starts:

- the drone is placed farther back and near ground level

For other gates:

- the drone is placed with wider lateral variation
- altitude is randomized more broadly

This creates a more diverse training set of approaches.

### Transform local spawn to world coordinates
The local offsets are rotated by the gate yaw so that the spawn position is expressed in the world frame.

Code reference: [quadcopter_strategies.py:374](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:374) through [quadcopter_strategies.py:390](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:374)

This is an important detail: the drone is not just dropped randomly in world coordinates. It is spawned relative to the selected gate, which makes the training starts semantically meaningful.

### Initial orientation
Once the spawn position is chosen, the drone is pointed roughly toward the gate center with a bit of yaw jitter.

Code reference: [quadcopter_strategies.py:392](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:392)

Why this helps:

- it prevents the task from being unnecessarily impossible at reset
- it still preserves variation, so the policy does not memorize one exact initial pose

### Evaluation behavior
When `is_train` is false, the reset behavior is simpler and more controlled.

The play/evaluation reset keeps the drone near the start area so rollouts are easier to inspect visually.

Code reference: [quadcopter_strategies.py:402](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:402)

## Design Philosophy Summary
The current implementation is trying to balance three goals:

1. Geometric correctness
The gate logic is based on real plane crossing and gate-window containment, not distance heuristics.

2. Simplicity
The reward is intentionally slimmed down to a small number of terms.

3. Anti-exploitation
The code explicitly guards against a known failure mode: re-crossing a previously passed gate to cheat the waypoint logic.

## Tradeoffs
This design has some clear strengths:

- easier to reason about than a very dense reward
- more faithful gate logic
- better protection against shortcut behaviors
- observation space contains concrete geometric information

It also has some possible downsides:

- gate-corner observations are fairly high-dimensional
- the progress reward is still somewhat shaped, not purely sparse
- anti-cheat logic adds some statefulness that must be maintained carefully

## Practical Reading Order
If you want to understand the file quickly, read it in this order:

1. [get_rewards()](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:68)
2. [reset_idx()](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:283)
3. [get_observations()](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py:213)

That order tends to make the most sense conceptually:

- first define what counts as good behavior
- then define where training episodes begin
- then define what information the policy gets to act on
