# Heatmap script

This script is designed to process feature files and generate heatmaps using an exported learner using the ViT regression. The script takes three input arguments:

- `--learner_path`: Path to the exported learner (.pkl) file.
- `--feature_name_pattern`: Pattern to match feature files (e.g., '/path/to/files/*.h5').
- `--output_folder`: Path to the output folder where results will be saved.


### Example

\`\`\`bash
python transformer_heatmap.py \
    --learner_path "/path/to/export.pkl" \
    --feature_name_pattern "/path/to/slide/features/*.h5" \
    --output_folder "path/to/store/output"
\`\`\`

