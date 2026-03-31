# Quadcopter Reward Structure: Technical Breakdown

This document gives a mathematical description of the current observation and reward structure implemented in [quadcopter_strategies.py](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py).

It focuses on two questions:

1. What information is available to the policy and reward function?
2. How is that information transformed into the scalar reward used for learning?

## 1. Notation
Let the environment index be omitted for clarity.

### State and geometry
- Drone world position: `p_t \in \mathbb{R}^3`
- Drone position relative to the current gate frame: `g_t = [x_t, y_t, z_t]^T \in \mathbb{R}^3`
- Previous drone world position: `p_{t-1}`
- Previous drone position in the current gate frame: `g_{t-1} = [x_{t-1}, y_{t-1}, z_{t-1}]^T`
- Current gate index: `i_t`
- Next gate index: `j_t = (i_t + 1) mod N`
- Current target gate center in world frame: `w_{i_t}`
- Next gate center in world frame: `w_{j_t}`

### Stored helper state
The reward implementation also maintains a few auxiliary quantities:

- Last passed gate index: `\ell_t`
- Previous pose relative to the last passed gate: `\tilde g_{t-1}`
- Previous distance to current target gate: `d_{t-1}`

These are not part of the observation given to the policy, but they are part of the reward computation.

## 2. Observation Structure
Code reference: [quadcopter_strategies.py](/home/ethan/ese651_project/src/isaac_quad_sim2real/tasks/race/config/crazyflie/quadcopter_strategies.py)

The policy observation is a concatenation of four components.

### 2.1 Drone body-frame linear velocity
Let the drone body-frame linear velocity be

$$
v_t^{(b)} \in \mathbb{R}^3.
$$

This gives the policy direct access to translational motion in the local control frame.

### 2.2 Drone attitude matrix
The drone orientation is represented as a rotation matrix

$$
R_t \in SO(3),
$$

which is flattened into a 9-dimensional vector.

This is used instead of raw Euler angles in order to provide a smooth and explicit orientation representation.

### 2.3 Current gate corners in the drone body frame
Let the four corners of the current gate in world frame be

$$
C_t^{(1)}, C_t^{(2)}, C_t^{(3)}, C_t^{(4)} \in \mathbb{R}^3.
$$

These are transformed into the drone body frame:

$$
\hat C_t^{(k)} \in \mathbb{R}^3, \quad k = 1,2,3,4.
$$

These 4 points provide a geometric description of the current gate opening relative to the drone.

### 2.4 Next gate corners in the drone body frame
Similarly, the next gate corners are transformed to the body frame:

$$
\hat N_t^{(k)} \in \mathbb{R}^3, \quad k = 1,2,3,4.
$$

This gives the policy a one-step lookahead of the course geometry.

### 2.5 Final observation vector
The observation concatenation is therefore

$$
o_t = \Big[v_t^{(b)}, \; \mathrm{vec}(R_t), \; \mathrm{vec}(\hat C_t), \; \mathrm{vec}(\hat N_t)\Big].
$$

Dimension count:

- body velocity: 3
- attitude matrix: 9
- current gate corners: 12
- next gate corners: 12

Total observation dimension:

$$
3 + 9 + 12 + 12 = 36.
$$

## 3. Reward Structure Overview
The reward used in training is

$$
r_t = r_t^{\text{progress}} + r_t^{\text{crash}},
$$

with an override to `death_cost` when the environment terminates.

In code, these two named heads are:

- `progress_goal`
- `crash`

The reward scales currently defined in [train_race.py](/home/ethan/ese651_project/scripts/rsl_rl/train_race.py) are:

$$
\lambda_{\text{progress}} = 50.0,
$$
$$
\lambda_{\text{crash}} = -1.0,
$$
$$
\lambda_{\text{death}} = -10.0.
$$

So the actual scalar reward is

$$
r_t = \lambda_{\text{progress}} \cdot P_t + \lambda_{\text{crash}} \cdot C_t,
$$

unless the episode is terminated, in which case

$$
r_t = \lambda_{\text{death}}.
$$

The main technical content is therefore in the definition of `P_t` and `C_t`.

## 4. Valid Gate Passage Detection
A valid gate passage is not defined by simple proximity. It is defined geometrically.

### 4.1 Crossing the gate plane
The drone must move from the front side of the gate plane to the back side:

$$
\text{crossed\_plane}_t = (x_{t-1} > 0) \land (x_t \le 0).
$$

### 4.2 Interpolated crossing point
To estimate where the drone crossed the plane `x = 0`, define

$$
\alpha_t = \mathrm{clip}\left(\frac{-x_{t-1}}{x_t - x_{t-1}}, 0, 1\right).
$$

Then the interpolated crossing coordinates inside the gate frame are

$$
y_t^{\text{cross}} = y_{t-1} + \alpha_t (y_t - y_{t-1}),
$$
$$
z_t^{\text{cross}} = z_{t-1} + \alpha_t (z_t - z_{t-1}).
$$

### 4.3 Gate opening containment
The gate opening is modeled as a square with half-extent

$$
h_{\text{gate}} = 0.5.
$$

The crossing is inside the legal opening iff

$$
\text{inside\_gate}_t = (|y_t^{\text{cross}}| \le h_{\text{gate}}) \land (|z_t^{\text{cross}}| \le h_{\text{gate}}).
$$

### 4.4 Forward course motion
Let the vector from the current gate center to the next gate center be

$$
s_t = w_{j_t} - w_{i_t}.
$$

Normalize it:

$$
\hat s_t = \frac{s_t}{\|s_t\|}.
$$

Let the drone world displacement be

$$
\Delta p_t = p_t - p_{t-1}.
$$

Then we require positive motion in the course direction:

$$
\text{forward\_course\_motion}_t = (\Delta p_t \cdot \hat s_t) > 0.
$$

### 4.5 Final valid-pass condition
A legal pass is therefore

$$
\text{gate\_passed}_t = \text{crossed\_plane}_t \land \text{inside\_gate}_t \land \text{forward\_course\_motion}_t.
$$

## 5. Illegal Forward Crossing
A fast but illegal gate-plane crossing is explicitly penalized.

If the drone crosses the gate plane in the correct global direction but does not pass through the gate opening, this is treated as cheating/crashing.

Mathematically:

$$
\text{illegal\_forward\_cross}_t = \text{crossed\_plane}_t \land \text{forward\_course\_motion}_t \land \neg \text{inside\_gate}_t.
$$

This is the mechanism that discourages skipping corners and slicing past the gate frame without actually going through the opening.

## 6. Reverse Re-Crossing Cheating
The implementation also checks whether the drone goes back through the previously completed gate.

Let `\ell_t` be the last successfully passed gate index. The drone pose is transformed into that old gate frame. If it crosses from the back side to the front side of that gate while still inside the opening, this is flagged as cheating.

The reverse crossing condition is:

$$
\text{crossed\_back}_t = (\tilde x_{t-1} \le 0) \land (\tilde x_t > 0).
$$

Using the same interpolation logic as before, the implementation computes whether the reverse crossing occurred inside the previous gate opening. If so,

$$
\text{cheated}_t = 1.
$$

This prevents the policy from repeatedly farming the same gate.

## 7. Special Pair Arc Bonus
The current reward also contains a maneuver bonus designed for a specific class of gate pairs.

The idea is to encourage a high-line maneuver between two nearby gates when they:

- are close in the horizontal plane,
- share an axis,
- have similar height,
- and require essentially the same exit direction.

### 7.1 Special pair detection
For the last passed gate `\ell_t` and the current target gate `i_t`, define

$$
\delta_t = w_{i_t} - w_{\ell_t}.
$$

The pair is considered special if all of the following hold:

1. Close in the horizontal plane:

$$
\|\delta_{t,xy}\| < 1.75
$$

2. Similar exit direction:

$$
\cos(\psi_{i_t} - \psi_{\ell_t}) > 0.95
$$

3. Share an axis:

$$
\min(|\delta_{t,x}|, |\delta_{t,y}|) < 0.35
$$

4. Similar height:

$$
|\delta_{t,z}| < 0.35
$$

### 7.2 Between-gates condition
The drone must also be physically between the two gates when projected onto the line segment connecting them.

Let the 2D segment between the gate centers be

$$
q_t = w_{i_t,xy} - w_{\ell_t,xy}.
$$

Define the projection parameter

$$
\tau_t = \frac{(p_{t,xy} - w_{\ell_t,xy}) \cdot q_t}{\|q_t\|^2}.
$$

Then the drone is considered between the gates if

$$
0.1 < \tau_t < 0.9.
$$

### 7.3 Stay near the shared line
The closest point on the line segment is computed, and the lateral distance to the line is measured. The drone must remain near the line:

$$
\text{lateral\_dist}_t < 0.5.
$$

### 7.4 Height-over-line bonus
Let

$$
h_t^{\text{over}} = p_{t,z} - \max(w_{\ell_t,z}, w_{i_t,z}).
$$

Then define the positive high-line amount

$$
\text{high\_line}_t = \mathrm{clip}(h_t^{\text{over}} - 0.25, 0, 0.5).
$$

The maneuver bonus is

$$
A_t = \mathbf{1}_{\text{special pair}} \cdot \mathbf{1}_{0.1 < \tau_t < 0.9} \cdot \mathbf{1}_{\text{lateral\_dist}_t < 0.5} \cdot \text{high\_line}_t.
$$

This encourages the drone to take a higher line between those special gate pairs rather than flattening out or drifting around the side.

## 8. Progress Reward Construction
The core `progress_goal` quantity is composed of four parts.

### 8.1 Centering progress
The current and previous gate-frame lateral errors are

$$
e_t^{\text{center}} = \|(y_t, z_t)\|,
$$
$$
e_{t-1}^{\text{center}} = \|(y_{t-1}, z_{t-1})\|.
$$

Centering progress is

$$
P_t^{\text{center}} = \mathrm{clip}(e_{t-1}^{\text{center}} - e_t^{\text{center}}, 0, 0.03).
$$

This rewards getting more centered relative to the current gate opening.

### 8.2 Forward progress toward the target gate
Let the current target distance be

$$
d_t = \|w_{i_t} - p_t\|.
$$

Then forward progress is

$$
P_t^{\text{forward}} = \mathrm{clip}(d_{t-1} - d_t, 0, 0.15).
$$

### 8.3 Safe corridor gating
Forward progress is only rewarded when the drone is inside a tighter safe corridor inside the gate opening.

The safe corridor half-extent is

$$
h_{\text{safe}} = 0.35.
$$

The corridor indicator is

$$
\text{corridor}_t = (|y_t| \le h_{\text{safe}}) \land (|z_t| \le h_{\text{safe}}).
$$

So only corridor-aligned forward progress counts:

$$
\mathbf{1}_{\text{corridor}_t} P_t^{\text{forward}}.
$$

### 8.4 Gate-pass bonus
A legal gate pass contributes an additional fixed bonus:

$$
P_t^{\text{pass}} = 0.75 \cdot \mathbf{1}_{\text{gate\_passed}_t}.
$$

### 8.5 Arc bonus contribution
The special maneuver contribution enters as

$$
P_t^{\text{arc}} = 0.4 \cdot A_t.
$$

### 8.6 Total progress term
Putting everything together, the unscaled progress quantity is

$$
P_t = 0.2 P_t^{\text{center}} + \mathbf{1}_{\text{corridor}_t} P_t^{\text{forward}} + P_t^{\text{pass}} + P_t^{\text{arc}}.
$$

This is the quantity stored in the code as `progress`.

## 9. Crash Term
The crash head is built from two sources:

1. Physical collision from the contact sensor
2. Logical cheating / illegal crossing conditions

Let

$$
C_t^{\text{contact}} \in \{0,1\}
$$

be the contact-based crash indicator, and let

$$
C_t^{\text{logic}} = \max(\text{cheated}_t, \text{illegal\_forward\_cross}_t).
$$

Then the crash signal is

$$
C_t = \max(C_t^{\text{contact}}, C_t^{\text{logic}}).
$$

This means both physical crashes and reward-defined illegal maneuvers are treated consistently.

## 10. Final Training Reward
The final reward before termination override is

$$
r_t = 50.0 \cdot P_t - 1.0 \cdot C_t.
$$

If the environment is terminated, then the reward is replaced with

$$
r_t = -10.0.
$$

## 11. Interpretation
The current reward structure can be summarized as follows:

- The drone is rewarded for becoming more centered with respect to the gate.
- It is rewarded more strongly for making forward progress only when it is already reasonably aligned.
- A legal pass gives a discrete bonus.
- A specific high-line maneuver between certain gate pairs gives an extra bonus.
- Illegal crossings and reverse re-crossings are treated as crashes.

In short, the reward is not merely encouraging speed. It is encouraging:

1. legal traversal,
2. alignment before commitment,
3. forward progress through the course,
4. and, in special geometric cases, a preferred high-line maneuver.
