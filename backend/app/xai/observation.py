"""
Rule-based textual observation of Grad-CAM activation regions.
Feeds structured input to the LLM prompt rather than asking the LLM to interpret the image.
"""
import numpy as np


REGION_NAMES = {
    (0, 0): "upper-left (right lung apex)",
    (0, 1): "upper-center (mediastinum/trachea)",
    (0, 2): "upper-right (left lung apex)",
    (1, 0): "middle-left (right lung mid-zone)",
    (1, 1): "center (cardiac silhouette)",
    (1, 2): "middle-right (left lung mid-zone)",
    (2, 0): "lower-left (right lung base)",
    (2, 1): "lower-center (diaphragm)",
    (2, 2): "lower-right (left lung base)",
}


def generate_observation(cam: np.ndarray) -> str:
    """
    Analyze a Grad-CAM heatmap and produce a structured textual description
    of which lung regions show the highest model activation.
    """
    h, w = cam.shape
    row_step, col_step = h // 3, w // 3

    region_scores = {}
    for r in range(3):
        for c in range(3):
            r_start, r_end = r * row_step, (r + 1) * row_step if r < 2 else h
            c_start, c_end = c * col_step, (c + 1) * col_step if c < 2 else w
            region = cam[r_start:r_end, c_start:c_end]
            region_scores[(r, c)] = float(region.mean())

    # Sort by activation intensity, descending
    ranked = sorted(region_scores.items(), key=lambda x: x[1], reverse=True)

    max_activation = ranked[0][1]
    if max_activation < 0.1:
        return "The model shows minimal activation across all lung regions, suggesting low confidence in pathological findings."

    # Identify regions with significant activation (>50% of peak)
    threshold = max_activation * 0.5
    active_regions = [(pos, score) for pos, score in ranked if score >= threshold]

    parts = []
    for pos, score in active_regions[:3]:  # top 3 most active regions
        name = REGION_NAMES[pos]
        intensity = "strong" if score > 0.7 else "moderate" if score > 0.4 else "mild"
        parts.append(f"{intensity} activation in the {name} region (intensity: {score:.2f})")

    primary = parts[0] if parts else "no significant activation detected"
    secondary = "; ".join(parts[1:]) if len(parts) > 1 else "no additional regions of note"

    laterality = _determine_laterality(active_regions)

    return (
        f"Primary finding: {primary}. "
        f"Additional regions: {secondary}. "
        f"Pattern: {laterality}."
    )


def _determine_laterality(active_regions: list) -> str:
    left_score = sum(s for (r, c), s in active_regions if c == 0)
    right_score = sum(s for (r, c), s in active_regions if c == 2)
    center_score = sum(s for (r, c), s in active_regions if c == 1)

    if left_score > right_score * 1.5:
        return "predominantly right-lung involvement (radiographic left)"
    elif right_score > left_score * 1.5:
        return "predominantly left-lung involvement (radiographic right)"
    elif center_score > (left_score + right_score):
        return "central/mediastinal pattern"
    else:
        return "bilateral or diffuse pattern"
