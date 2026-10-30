# Annotation guideline

Label every comment as **Sarcastic** or **Sincere**, using only the text shown.

## Sarcastic
The writer says the opposite of what they mean, to mock, criticise or joke.
- Praise words used for a bad situation: "wah kya baat hai, bilkul time pe aaye"
- Exaggerated agreement or fake admiration of a person/group
- Rhetorical or mocking questions that imply the opposite
- Positive words + a clearly negative situation: "great, another Monday meeting"

## Sincere
The words match the intent: genuine praise, thanks, complaint, opinion, question, or plain information.
- Abuse or insults are **Sincere** unless irony is clear (anger is not sarcasm).
- Jokes without a reversal of meaning are **Sincere**.

## Rules
1. Judge the comment alone; do not look up the video or post.
2. Emojis (written as words like `face_with_tears_of_joy`) count as tone signals but are not enough alone.
3. If you truly cannot decide, pick the label you think is more likely and mark the row in a `notes` column.
4. Label independently. Do not see each other's labels until both finish.

## Workflow
1. `python -m src.make_annotation_sample` creates `reports/double_annotation_template.csv`.
2. Each person fills only their own column (`khushi`, `grishma`) with `Sarcastic` or `Sincere`.
3. Save the combined file as `reports/double_annotation_completed.csv`.
4. `python -m src.make_annotation_sample --kappa` prints agreement and Cohen's kappa and saves disagreements for discussion.
