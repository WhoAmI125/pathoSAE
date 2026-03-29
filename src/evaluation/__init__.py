from src.evaluation.compare import compare_checkpoints
from src.evaluation.feature_analysis import (
    analyze_checkpoint_features,
    analyze_features,
    iter_activation_batches,
    resolve_activation_files_for_split,
)
from src.evaluation.feature_figures import (
    build_feature_dataframe,
    plot_feature_entropy_hist,
    plot_feature_scatter,
    plot_top_feature_reference_grid,
    save_top_feature_reference_images_individual,
)
from src.evaluation.metrics import (
    EvaluationArtifacts,
    compute_reconstruction_metrics,
    evaluate_checkpoint,
    save_metrics,
)
from src.evaluation.visualize import plot_activation_histogram, plot_sparsity_distribution

# Optional import: group analysis depends on seaborn/scipy stack, which may not be
# available in every runtime where basic evaluation is used.
try:
    from src.evaluation.group_analysis import (
        analyze_groups_for_checkpoint,
        load_meta_csv,
        match_meta_record,
        plot_group_heatmap,
    )
except Exception:  # pragma: no cover
    analyze_groups_for_checkpoint = None
    load_meta_csv = None
    match_meta_record = None
    plot_group_heatmap = None

__all__ = [
    "EvaluationArtifacts",
    "compute_reconstruction_metrics",
    "evaluate_checkpoint",
    "save_metrics",
    "analyze_features",
    "analyze_checkpoint_features",
    "iter_activation_batches",
    "resolve_activation_files_for_split",
    "plot_activation_histogram",
    "plot_sparsity_distribution",
    "build_feature_dataframe",
    "plot_feature_scatter",
    "plot_feature_entropy_hist",
    "plot_top_feature_reference_grid",
    "save_top_feature_reference_images_individual",
    "load_meta_csv",
    "match_meta_record",
    "plot_group_heatmap",
    "analyze_groups_for_checkpoint",
    "compare_checkpoints",
]
