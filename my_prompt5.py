def process_image_ego_prompt(action, object_name):
    return f"""
        You are given an image showing a '{object_name}' involved in the action '{action}'.

        🎯 Your task:
        Identify several **precise keypoints** in the image that are essential for performing the action '{action}' on the object '{object_name}'.

        ⚠️ Important Instructions:
        - Only return **single-point** coordinates in the format [x, y]
        - Do **not** return bounding boxes or regions
        - All points must lie **within** the '{object_name}'
        - Avoid placing multiple points too close together
        - ❌ Do **not** include any text, comments, or labels

        ✅ Output format (strict):
        [
        [x1, y1],
        [x2, y2],
        [x3, y3]
        ]
        """

def process_image_ego_prompt_w_pred(action, object_name):
    return f"""
        You are given an image showing a '{object_name}' involved in the action '{action}'.

        🎯 Your task:
        Identify several **precise keypoints** in the image that are essential for performing the action '{action}' on the object '{object_name}'.
        and rating the importance of that point

        ⚠️ Important Instructions:
        - Only return **single-point** coordinates and prediction score in the format [x, y, prediction_score]
        - Do **not** return bounding boxes or regions
        - All points must lie **within** the '{object_name}'
        - Avoid placing multiple points too close together
        - If there are more than one '{object_name}', give me point from each '{object_name}'
        - ❌ Do **not** include any text, comments, or labels

        ✅ Output format (strict):
        [
        [x1, y1,prediction_score],
        [x2, y2,prediction_score],
        [x3, y3,prediction_score]
        ]
        """

def process_image_exo_prompt(action, object_name):
    return f"""
    You are given two images:
    1. An **egocentric** image where you must select keypoints.
    2. An **exocentric** reference image showing how the action '{action}' is typically performed on the '{object_name}'.

    🎯 Task:
    Select multiple [x, y] keypoints in the **egocentric image** that are critical for performing the action '{action}' on the '{object_name}'.

    🔍 Use the exocentric image to:
    - Understand typical interaction patterns
    - Identify functionally important parts (e.g., contact or force areas)

    📌 Guidelines:
    - Keypoints must lie **within** the '{object_name}' in the egocentric image
    - If there are multiple '{object_name}' instances, mark keypoints on **each of them**
    - Place **at least 3 well-separated** points covering the entire functional region
    - e.g., for a handle: both ends and the center
    - Avoid clustering or irrelevant placements

    ⛔ Do NOT:
    - Include text, labels, bounding boxes, or extra formatting

    ✅ Output format (strict):
    [
    [x1, y1],
    [x2, y2],
    [x3, y3]
    ]
    """

def process_image_exo_prompt_w_pred(action, object_name):
    return f"""
    You are given two images:
    1. An **egocentric** image where you must select keypoints.
    2. An **exocentric** reference image showing how the action '{action}' is typically performed on the '{object_name}'.

    🎯 Task:
    Select multiple [x, y] keypoints in the **egocentric image** that are critical for performing the action '{action}' on the '{object_name}'.

    🔍 Use the exocentric image to:
    - Understand typical interaction patterns
    - Identify functionally important parts (e.g., contact or force areas)

    📌 Guidelines:
    - Keypoints must lie **within** the '{object_name}' in the egocentric image
    - If there are multiple '{object_name}' instances, mark keypoints on **each of them**
    - Place **at least 3 well-separated** points covering the entire functional region
    - If there are more than one '{object_name}', give me point from each '{object_name}'
    - Avoid clustering or irrelevant placements

    ⛔ Do NOT:
    - Include text, labels, bounding boxes, or extra formatting

    ✅ Output format (strict):
    [
    [x1, y1,prediction_score],
    [x2, y2,prediction_score],
    [x3, y3,prediction_score]
    ]
    """
