
def step1_w_exo(action, object_name):
    return f"""
    # MISSION: GENERATOR AGENT

You are the **Generator Agent**. Your task is to analyze the provided images (egocentric and exocentric) and the task description. Generate a **dense cloud of all plausible keypoints** in the **egocentric image** where the action could occur. Be exhaustive and do not filter or judge the points yet.

---

## INPUTS

* **Egocentric Image:** [Primary image for annotation]
* **Exocentric Image:** [Reference image for context]
* **Task:** Perform the action '{action}' on the '{object_name}'.

---

## OUTPUT FORMAT (Strict JSON)

Provide your answer inside a single JSON block with the key "generated_points".

```json
{
"generated_points": [
    [x1, y1],
    [x2, y2],
    [x3, y3]
]
}


"""

def step2(action, object_name):
    return f"""
# MISSION: INTERACTION PLANNER AGENT

You are the **Interaction Planner Agent**. Your task is to analyze the user's intended **Task** and the **exocentric reference image** to determine the optimal **Affordance Topology**. The topology describes the geometric 'shape' of the interaction.

---

## INPUTS

* **Exocentric Image:** [Reference image for context]
* **Task:** Perform the action '{action}' on the '{object_name}'.

---

## INSTRUCTIONS

1.  **Reasoning:** First, briefly explain your reasoning. What is the physical goal of the action '{action}'? Does it require precise force (`POINT`), interaction along an edge (`LINE/CURVE`), or convenience across an area (`REGION`)?
2.  **Decision:** Then, choose one of the three topology types.

---

## OUTPUT FORMAT (Strict JSON)

Provide your answer inside a single JSON block with the key "topology_analysis".

```json
{
  "topology_analysis": {
    "rationale": "The action 'open refrigerator' prioritizes convenience, allowing the user to pull from anywhere on the handle. Therefore, the interaction space is a functionally equivalent area.",
    "topology": "REGION"
  }
}

    """

def step3(action, object_name):
    return f"""
# MISSION: REFINER AGENT

You are the **Refiner Agent**. Your task is to take the dense point cloud from the Generator and the strategic topology decision from the Planner to produce the final, precise set of keypoints.

---

## INPUTS

* **Generated Points:** A list of all plausible keypoints, e.g., `[[x1, y1], [x2, y2], ...]`
* **Topology Analysis:** The decision from the planner, e.g., `{"topology": "REGION", "rationale": "..."}`

---

## INSTRUCTIONS

Apply the topology decision to filter the generated points using the following logic:

* **If `topology` is `POINT`:** Select the **single most optimal point** from the candidates that best achieves the action's goal.
* **If `topology` is `LINE/CURVE`:** Select the subset of points that form the most relevant **continuous line or curve**. Discard outliers.
* **If `topology` is `REGION`:** Select **all points** that fall within the primary functional area. Discard any stray points outside this main cluster.

---

## OUTPUT FORMAT (Strict JSON)

Provide your final answer inside a single JSON block with the key "final_keypoints". The output should only contain the list of coordinates.

```json
{
  "final_keypoints": [
    [x_final_1, y_final_1],
    [x_final_2, y_final_2]
  ]
}

    """