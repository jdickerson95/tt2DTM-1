"""Extracting results to DataFrame."""

from leopard_em.pydantic_models.managers import MatchTemplateManager

# Update this path based on which match template config you want to use
yaml_path = "match_template_config_example.yaml"
false_positive_rate = 100
total_corr = int(13 * 1.59e6)
DATAFRAME_OUTPUT_PATH = f"match_template_results_false-positive-rate-{false_positive_rate}.csv"


# Instantiate the MatchTemplateManager from the config and get the result object
mt_manager = MatchTemplateManager.from_yaml(yaml_path)
mt_result = mt_manager.match_template_result
mt_result.load_tensors_from_paths()  # Needed to load results into memory

# Manually set the number of correlations; used for z-score cutoff determination
# Is automatically calculated after an actual run
mt_result.total_projections = total_corr
df_full = mt_manager.results_to_dataframe(locate_peaks_kwargs={"false_positives": false_positive_rate})
df_full.to_csv(DATAFRAME_OUTPUT_PATH, index=True)
