"""
FeatureCraft Feature Engineering Pipeline Flowchart Generator

This script generates a visual flowchart of the FeatureCraft automated feature engineering 
pipeline showing separate paths for REGRESSION and CLASSIFICATION datasets with shared steps.

FEATURES VISUALIZED:
    - Schema Validation (Pandera) & Leakage Prevention
    - Multi-strategy Encoding (OHE, Target, Frequency, Count, WoE, Hashing, Ordinal)
    - Out-of-Fold Target Encoding with CV strategies (KFold, Stratified, Group, TimeSeries)
    - SMOTE & Class Imbalance Handling
    - Drift Detection (PSI, KS Test)
    - Feature Selection (Correlation, VIF, Mutual Info, WoE/IV)
    - Dimensionality Reduction (PCA, SVD, UMAP)
    - Text Processing (TF-IDF, Hashing, SVD)
    - DateTime Features (Cyclic, Fourier, Holiday)

INSTALLATION:
    pip install graphviz networkx matplotlib

USAGE:
    python feature_flow.py

CONFIGURATION:
    Edit FLOW_CONFIG["highlight_dataset_type"] to emphasize:
    - "regression" : highlight regression path
    - "classification" : highlight classification path
    - "both" : highlight both paths equally

OUTPUT:
    - Graphviz (if available): feature_engineering_flow.svg (auto-opens)
    - Fallback: feature_engineering_flow.png (matplotlib window)
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# Try Graphviz first, fallback to NetworkX + Matplotlib
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("⚠ Graphviz not available, will use NetworkX fallback", file=sys.stderr)

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# CONFIGURATION - BASED ON ACTUAL FEATURECRAFT LIBRARY
# ============================================================================

FLOW_CONFIG: Dict[str, Any] = {
    "library_name": "FeatureCraft AutoML Pipeline",
    "dataset_types": ["regression", "classification"],
    "highlight_dataset_type": "both",  # "regression", "classification", or "both"
    
    "stages": {
        "Ingestion & Analysis": {
            "common": [
                {"id": "LOAD_DATA", "label": "Load Dataset"},
                {"id": "VALIDATE_INPUT", "label": "Validate Input\n(Types, Shape)"},
                {"id": "SPLIT_FEATURES_TARGET", "label": "Split Features/Target"},
                {"id": "INFER_PROBLEM_TYPE", "label": "Auto-Detect Task Type\n(>15 unique = Regression)"},
                {"id": "PROFILE_COLUMNS", "label": "Profile Columns\n(Cardinality, Missing, Skew)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Schema Validation": {
            "common": [
                {"id": "SCHEMA_VALIDATE", "label": "Schema Validation\n(Pandera: types, drift, ranges)"},
                {"id": "SCHEMA_COERCE", "label": "Type Coercion\n(Optional: auto-fix)"},
                {"id": "LEAKAGE_GUARD", "label": "Leakage Prevention\n(Block target in transform)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Column Type Detection": {
            "common": [
                {"id": "DETECT_NUMERIC", "label": "Detect Numeric Columns\n(pd.to_numeric validation)"},
                {"id": "DETECT_CATEGORICAL", "label": "Detect Categorical\n(Low/Mid/High Cardinality)"},
                {"id": "DETECT_DATETIME", "label": "Detect DateTime Columns"},
                {"id": "DETECT_TEXT", "label": "Detect Text Columns\n(avg length ≥15)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Numeric Pipeline": {
            "common": [
                {"id": "NUMERIC_CONVERT", "label": "NumericConverter\n(Coerce to numeric)"},
                {"id": "NUMERIC_IMPUTE", "label": "Numeric Imputation\n(Simple/Advanced)"},
                {"id": "SKEW_TRANSFORM", "label": "Power Transform\n(Yeo-Johnson if |skew|>1)"},
                {"id": "WINSORIZE", "label": "Winsorization\n(Optional: clip outliers)"},
            ],
            "regression_only": [
                {"id": "ROBUST_SCALING_REG", "label": "Robust Scaling\n(If heavy outliers)"},
            ],
            "classification_only": []
        },
        "Categorical Pipeline": {
            "common": [],
            "regression_only": [],
            "classification_only": []
        },
        "Low Cardinality (≤10)": {
            "common": [
                {"id": "CAT_LOW_RARE", "label": "Rare Category Grouping\n(freq < 1% → 'Other')"},
                {"id": "CAT_LOW_IMPUTE", "label": "Categorical Imputation\n(Most Frequent)"},
                {"id": "CAT_LOW_OHE", "label": "One-Hot Encoding"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Mid Cardinality (11-50)": {
            "common": [
                {"id": "CAT_MID_IMPUTE", "label": "Categorical Imputation"},
            ],
            "regression_only": [
                {"id": "CAT_MID_FREQ", "label": "Frequency Encoding\n(Optional: category → freq)"},
                {"id": "CAT_MID_COUNT", "label": "Count Encoding\n(Optional: category → count)"},
                {"id": "CAT_MID_HASH_FALLBACK", "label": "Hashing Encoder\n(Fallback if TE disabled)"},
            ],
            "classification_only": [
                {"id": "CAT_MID_TE_OOF", "label": "Out-of-Fold Target Encoding\n(KFold/Stratified/Group CV-safe)"},
                {"id": "CAT_MID_TE_LOO", "label": "Leave-One-Out Target Encoding\n(Alternative to OOF)"},
                {"id": "CAT_MID_WOE", "label": "Weight of Evidence\n(Binary classification only)"},
            ]
        },
        "High Cardinality (>50)": {
            "common": [
                {"id": "CAT_HIGH_IMPUTE", "label": "Categorical Imputation"},
                {"id": "CAT_HIGH_RARE", "label": "Rare Category Grouping"},
                {"id": "CAT_HIGH_HASH", "label": "Hashing Encoder\n(256 features)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Text Features": {
            "common": [
                {"id": "TEXT_SELECT", "label": "Text Column Selector"},
                {"id": "TEXT_VECTORIZE", "label": "TF-IDF / Hashing\n(max_features=20k)"},
                {"id": "TEXT_SVD", "label": "SVD Reduction\n(Optional: trees only)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "DateTime Features": {
            "common": [
                {"id": "DT_EXTRACT", "label": "DateTime Extraction\n(Year, Month, Day, DoW, Hour)"},
                {"id": "DT_CYCLIC", "label": "Cyclic Encoding\n(Sin/Cos for month, weekday)"},
                {"id": "DT_FOURIER", "label": "Fourier Features\n(Optional: orders 3,7)"},
                {"id": "DT_HOLIDAY", "label": "Holiday Features\n(Optional: country-specific)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Scaling & Transform": {
            "common": [],
            "regression_only": [
                {"id": "SCALE_LINEAR", "label": "Standard Scaling\n(Linear/SVM models)"},
            ],
            "classification_only": [
                {"id": "SCALE_MINMAX", "label": "MinMax Scaling\n(KNN/NN models)"},
            ]
        },
        "Class Imbalance": {
            "common": [],
            "regression_only": [],
            "classification_only": [
                {"id": "DETECT_IMBALANCE", "label": "Detect Minority Ratio"},
                {"id": "SMOTE", "label": "SMOTE Oversampling\n(If ratio < 10%)"},
                {"id": "CLASS_WEIGHTS", "label": "Class Weight Advisory\n(If ratio < 20%)"},
            ]
        },
        "Post-Processing": {
            "common": [
                {"id": "COLUMN_TRANSFORMER", "label": "ColumnTransformer\n(Combine all pipelines)"},
                {"id": "DIMENSIONALITY_REDUCE", "label": "Dimensionality Reduction\n(Optional: PCA/SVD/UMAP)"},
                {"id": "ENSURE_NUMERIC", "label": "Ensure Numeric Output\n(Final validation)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Feature Selection": {
            "common": [
                {"id": "CORR_PRUNE", "label": "Correlation Pruning\n(threshold=0.95)"},
                {"id": "VIF_CHECK", "label": "VIF Multicollinearity\n(threshold=10)"},
                {"id": "MUTUAL_INFO", "label": "Mutual Information\n(Optional: top K)"},
            ],
            "regression_only": [],
            "classification_only": [
                {"id": "WOE_IV_SELECT", "label": "WoE/IV Selection\n(IV threshold=0.02)"},
            ]
        },
        "Drift Detection": {
            "common": [
                {"id": "DRIFT_PSI", "label": "PSI (Categorical)\n(threshold=0.25)"},
                {"id": "DRIFT_KS", "label": "KS Test (Numeric)\n(threshold=0.10)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
        "Materialization": {
            "common": [
                {"id": "EXPORT_PIPELINE", "label": "Export Pipeline\n(pipeline.joblib)"},
                {"id": "EXPORT_METADATA", "label": "Export Metadata\n(metadata.json)"},
                {"id": "EXPORT_FEATURES", "label": "Export Feature Names\n(feature_names.txt)"},
                {"id": "TRANSFORM_OUTPUT", "label": "Transform & Output\n(Train/Valid DataFrames)"},
            ],
            "regression_only": [],
            "classification_only": []
        },
    },
    
    "custom_steps": [
        {
            "id": "CAT_CLEANER",
            "label": "Categorical Cleaner\n(Lowercase, Strip, Normalize)",
            "stage": "Column Type Detection",
            "applies_to": ["both"],
            "depends_on": ["DETECT_CATEGORICAL"],
            "notes": "Clean categorical data before encoding"
        },
        {
            "id": "MISSING_INDICATORS",
            "label": "Missing Indicators\n(If missing > 5%)",
            "stage": "Numeric Pipeline",
            "applies_to": ["both"],
            "depends_on": ["NUMERIC_IMPUTE"],
            "notes": "Binary flags for missingness"
        },
        {
            "id": "ORDINAL_ENCODE",
            "label": "Ordinal Encoding\n(Manual category ordering)",
            "stage": "Mid Cardinality (11-50)",
            "applies_to": ["both"],
            "depends_on": ["CAT_MID_IMPUTE"],
            "notes": "For ordered categorical features"
        },
        {
            "id": "WINSORIZER",
            "label": "Winsorization\n(Clip percentiles: 1%-99%)",
            "stage": "Numeric Pipeline",
            "applies_to": ["both"],
            "depends_on": ["NUMERIC_IMPUTE"],
            "notes": "Clip extreme outliers if enabled"
        },
    ],
    
    "style": {
        "common_shape": "box",
        "regression_shape": "box",  # will have rounded style
        "classification_shape": "box",  # will have double border
        "show_legend": True,
        "rankdir": "LR"
    }
}


# ============================================================================
# GRAPH BUILDER
# ============================================================================

@dataclass
class Node:
    id: str
    label: str
    stage: str
    applies_to: List[str]  # ["both"], ["regression"], ["classification"]
    shape: str = "box"
    style: str = ""
    
    
def build_dag(config: Dict[str, Any]) -> nx.DiGraph:
    """Build a NetworkX DAG from configuration."""
    G = nx.DiGraph()
    node_registry: Dict[str, Node] = {}
    stage_order = list(config["stages"].keys())
    
    # Build nodes from stages
    for stage_idx, (stage_name, stage_steps) in enumerate(config["stages"].items()):
        prev_common = None
        
        # Common steps
        for step in stage_steps.get("common", []):
            node = Node(
                id=step["id"],
                label=step["label"],
                stage=stage_name,
                applies_to=["both"],
                shape="box",
                style=""
            )
            node_registry[node.id] = node
            G.add_node(node.id, **node.__dict__)
            
            # Connect to previous common node in this stage or previous stage
            if prev_common:
                G.add_edge(prev_common, node.id)
            elif stage_idx > 0:
                # Connect to last common node of previous stage
                prev_stage_common = [n for n in G.nodes() 
                                     if G.nodes[n]["stage"] == stage_order[stage_idx - 1] 
                                     and "both" in G.nodes[n]["applies_to"]]
                if prev_stage_common:
                    G.add_edge(prev_stage_common[-1], node.id)
            
            prev_common = node.id
        
        # Regression-only steps
        for step in stage_steps.get("regression_only", []):
            node = Node(
                id=step["id"],
                label=step["label"],
                stage=stage_name,
                applies_to=["regression"],
                shape="box",
                style="rounded"
            )
            node_registry[node.id] = node
            G.add_node(node.id, **node.__dict__)
            
            # Connect to last common node in this stage
            if prev_common:
                G.add_edge(prev_common, node.id)
        
        # Classification-only steps
        for step in stage_steps.get("classification_only", []):
            node = Node(
                id=step["id"],
                label=step["label"],
                stage=stage_name,
                applies_to=["classification"],
                shape="box",
                style="double"
            )
            node_registry[node.id] = node
            G.add_node(node.id, **node.__dict__)
            
            # Connect to last common node in this stage
            if prev_common:
                G.add_edge(prev_common, node.id)
    
    # Add custom steps
    for custom in config.get("custom_steps", []):
        applies = custom["applies_to"]
        if "both" in applies:
            applies_list = ["both"]
        else:
            applies_list = applies
            
        style = ""
        if "regression" in applies and "classification" not in applies:
            style = "rounded"
        elif "classification" in applies and "regression" not in applies:
            style = "double"
            
        node = Node(
            id=custom["id"],
            label=custom["label"],
            stage=custom["stage"],
            applies_to=applies_list,
            shape="box",
            style=style
        )
        node_registry[node.id] = node
        G.add_node(node.id, **node.__dict__)
        
        # Wire dependencies
        if custom.get("depends_on"):
            for dep in custom["depends_on"]:
                if dep in G.nodes():
                    G.add_edge(dep, custom["id"])
    
    # Connect branch-specific nodes forward to next stage
    for node_id in list(G.nodes()):
        node_stage = G.nodes[node_id]["stage"]
        node_applies = G.nodes[node_id]["applies_to"]
        
        # Only process branch-specific nodes
        if node_applies == ["both"]:
            continue
            
        # Find if this node has outgoing edges to next stage
        out_edges = list(G.successors(node_id))
        if not out_edges:
            # No outgoing edges - need to connect to next stage
            stage_idx = stage_order.index(node_stage)
            if stage_idx + 1 < len(stage_order):
                next_stage = stage_order[stage_idx + 1]
                next_common = [n for n in G.nodes() 
                              if G.nodes[n]["stage"] == next_stage 
                              and "both" in G.nodes[n]["applies_to"]]
                if next_common:
                    G.add_edge(node_id, next_common[0])
    
    return G


# ============================================================================
# GRAPHVIZ RENDERER
# ============================================================================

def render_with_graphviz(G: nx.DiGraph, config: Dict[str, Any], output_path: Path) -> bool:
    """Render using Graphviz with clusters."""
    try:
        from graphviz import Digraph
    except ImportError:
        return False
    
    try:
        dot = Digraph(comment=config["library_name"])
        dot.attr(rankdir=config["style"]["rankdir"])
        dot.attr("node", fontname="Arial", fontsize="9")
        dot.attr("edge", fontname="Arial", fontsize="7")
        dot.attr(splines="ortho")  # Orthogonal routing for cleaner flow
        
        highlight = config["highlight_dataset_type"]
        stage_order = list(config["stages"].keys())
        
        # Create clusters per stage
        for stage_idx, stage_name in enumerate(stage_order):
            with dot.subgraph(name=f"cluster_{stage_idx}") as cluster:
                cluster.attr(label=stage_name, style="rounded", color="lightgrey", 
                           fontsize="11", fontname="Arial Bold")
                
                stage_nodes = [n for n in G.nodes() if G.nodes[n]["stage"] == stage_name]
                for node_id in stage_nodes:
                    node_data = G.nodes[node_id]
                    applies = node_data["applies_to"]
                    
                    # Determine style
                    node_style = []
                    color = "black"
                    fillcolor = "white"
                    
                    if node_data["style"] == "rounded":
                        node_style.append("rounded,filled")
                        fillcolor = "lightgreen"
                    elif node_data["style"] == "double":
                        node_style.append("filled")
                        fillcolor = "lightcoral"
                        cluster.node(node_id, node_data["label"], 
                                   shape=node_data["shape"], 
                                   style=",".join(node_style) if node_style else "filled",
                                   peripheries="2",
                                   fillcolor=fillcolor,
                                   color=color)
                        continue
                    else:
                        node_style.append("filled")
                        fillcolor = "lightblue"
                    
                    # Highlight based on config
                    penwidth = "1"
                    if highlight == "both":
                        penwidth = "2"
                    elif highlight == "regression" and "regression" in applies:
                        penwidth = "3"
                        color = "darkgreen"
                    elif highlight == "classification" and "classification" in applies:
                        penwidth = "3"
                        color = "darkred"
                    elif "both" in applies:
                        penwidth = "2"
                    
                    cluster.node(node_id, node_data["label"], 
                               shape=node_data["shape"],
                               style=",".join(node_style) if node_style else "filled",
                               penwidth=penwidth,
                               color=color,
                               fillcolor=fillcolor)
        
        # Add edges
        for u, v in G.edges():
            u_applies = G.nodes[u]["applies_to"]
            v_applies = G.nodes[v]["applies_to"]
            
            # Edge styling based on path
            penwidth = "1"
            color = "gray"
            
            if highlight == "both":
                penwidth = "2"
            elif highlight == "regression":
                if "regression" in u_applies or "regression" in v_applies:
                    penwidth = "2"
                    color = "darkgreen"
                elif "both" in u_applies or "both" in v_applies:
                    penwidth = "1.5"
            elif highlight == "classification":
                if "classification" in u_applies or "classification" in v_applies:
                    penwidth = "2"
                    color = "darkred"
                elif "both" in u_applies or "both" in v_applies:
                    penwidth = "1.5"
            
            dot.edge(u, v, penwidth=penwidth, color=color)
        
        # Add legend
        if config["style"].get("show_legend", True):
            with dot.subgraph(name="cluster_legend") as legend:
                legend.attr(label="Legend", style="dashed", color="grey", fontsize="10")
                legend.node("legend_common", "Common Step", shape="box", 
                          style="filled", fillcolor="lightblue")
                legend.node("legend_regression", "Regression Only", shape="box", 
                          style="rounded,filled", fillcolor="lightgreen")
                legend.node("legend_classification", "Classification Only", shape="box", 
                          peripheries="2", style="filled", fillcolor="lightcoral")
        
        # Render
        output_str = str(output_path.with_suffix(""))
        dot.render(output_str, format="svg", view=True, cleanup=True)
        print(f"✓ Graphviz render complete: {output_path}")
        return True
        
    except Exception as e:
        print(f"⚠ Graphviz rendering failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# NETWORKX FALLBACK RENDERER
# ============================================================================

def render_with_networkx(G: nx.DiGraph, config: Dict[str, Any], output_path: Path):
    """Fallback rendering with NetworkX + Matplotlib."""
    plt.figure(figsize=(24, 16))
    
    # Compute hierarchical layout
    try:
        # Try to use graphviz layout if available
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot", args="-Grankdir=LR")
    except:
        # Fallback to manual layered layout
        try:
            layers = list(nx.topological_generations(G))
            pos = {}
            layer_spacing = 4
            node_spacing = 2.5
            
            for layer_idx, layer in enumerate(layers):
                y_offset = -(len(layer) - 1) * node_spacing / 2
                for node_idx, node in enumerate(sorted(layer)):
                    pos[node] = (layer_idx * layer_spacing, y_offset + node_idx * node_spacing)
        except:
            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    highlight = config["highlight_dataset_type"]
    
    # Separate nodes by type
    common_nodes = [n for n in G.nodes() if "both" in G.nodes[n]["applies_to"]]
    regression_nodes = [n for n in G.nodes() if G.nodes[n]["applies_to"] == ["regression"]]
    classification_nodes = [n for n in G.nodes() if G.nodes[n]["applies_to"] == ["classification"]]
    
    # Draw edges first
    for u, v in G.edges():
        u_applies = G.nodes[u]["applies_to"]
        v_applies = G.nodes[v]["applies_to"]
        
        width = 1.5
        alpha = 0.5
        edge_color = "gray"
        
        if highlight == "both":
            width = 2
            alpha = 0.7
        elif highlight == "regression":
            if "regression" in u_applies or "regression" in v_applies:
                width = 3
                alpha = 0.9
                edge_color = "darkgreen"
            elif "both" in u_applies or "both" in v_applies:
                width = 2
                alpha = 0.7
        elif highlight == "classification":
            if "classification" in u_applies or "classification" in v_applies:
                width = 3
                alpha = 0.9
                edge_color = "darkred"
            elif "both" in u_applies or "both" in v_applies:
                width = 2
                alpha = 0.7
        
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, alpha=alpha, 
                              arrows=True, arrowsize=20, edge_color=edge_color, 
                              arrowstyle="-|>")
    
    # Draw nodes
    node_size = 3500
    
    if common_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=common_nodes, node_color="lightblue", 
                              node_size=node_size, node_shape="s", edgecolors="black", linewidths=2)
    
    if regression_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=regression_nodes, node_color="lightgreen",
                              node_size=node_size, node_shape="o", edgecolors="darkgreen", linewidths=2.5)
    
    if classification_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=classification_nodes, node_color="lightcoral",
                              node_size=node_size, node_shape="D", edgecolors="darkred", linewidths=3)
    
    # Draw labels
    labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold", font_family="sans-serif")
    
    # Legend
    if config["style"].get("show_legend", True):
        legend_elements = [
            mpatches.Patch(facecolor="lightblue", edgecolor="black", label="Common Step"),
            mpatches.Patch(facecolor="lightgreen", edgecolor="darkgreen", label="Regression Only"),
            mpatches.Patch(facecolor="lightcoral", edgecolor="darkred", label="Classification Only")
        ]
        plt.legend(handles=legend_elements, loc="upper left", fontsize=11, framealpha=0.9)
    
    plt.title(config["library_name"], fontsize=18, fontweight="bold", pad=20)
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"✓ NetworkX render complete: {output_path}")
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print(f"{'='*70}")
    print(f"  {FLOW_CONFIG['library_name']}")
    print(f"  Generating Feature Engineering Pipeline Flowchart")
    print(f"{'='*70}\n")
    
    # Build DAG
    print("Building DAG from FeatureCraft library structure...")
    G = build_dag(FLOW_CONFIG)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Highlighting: {FLOW_CONFIG['highlight_dataset_type']}\n")
    
    # Render
    if GRAPHVIZ_AVAILABLE:
        output_path = Path("feature_engineering_flow.svg")
        print("Attempting Graphviz render...")
        success = render_with_graphviz(G, FLOW_CONFIG, output_path)
        if not success:
            print("Falling back to NetworkX...\n")
            output_path = Path("feature_engineering_flow.png")
            render_with_networkx(G, FLOW_CONFIG, output_path)
    else:
        print("Using NetworkX fallback...\n")
        output_path = Path("feature_engineering_flow.png")
        render_with_networkx(G, FLOW_CONFIG, output_path)
    
    print(f"\n{'='*70}")
    print(f"  ✓ Flowchart generation complete!")
    print(f"  Output: {output_path.absolute()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

