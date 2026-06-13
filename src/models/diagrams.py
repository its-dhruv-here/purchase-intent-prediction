import os
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def generate_pipeline_diagram():
    dot_content = """digraph ML_Pipeline {
    rankdir=LR;
    node [shape=box, style="rounded,filled", fontname="Inter", fillcolor="#f8f9fa", color="#343a40", penwidth=1.5];
    edge [color="#495057", penwidth=1.2, fontname="Inter", fontsize=10];

    RawData [label="Raw Session Data\\n(CSV Format)", shape=cylinder, fillcolor="#e9ecef"];
    
    subgraph cluster_preprocessing {
        label = "Data Engineering & Preprocessing";
        style = "dashed";
        color = "#ced4da";
        fontname="Inter";
        
        Load [label="Data Loader\\n& Schema Validation"];
        NumScale [label="Numerical Scaling\\n(StandardScaler)"];
        CatEnc [label="Categorical Encoding\\n(OneHotEncoder)"];
        Split [label="Temporal/Stratified Split\\n(Train/Test)"];
        SMOTE [label="Class Balancing\\n(SMOTE Oversampling)"];
    }

    subgraph cluster_modeling {
        label = "Model Training & Evaluation";
        style = "dashed";
        color = "#ced4da";
        fontname="Inter";

        XGB [label="Primary Model\\n(XGBoost Classifier)", fillcolor="#d1e7dd", color="#0f5132"];
        RF [label="Challenger Model\\n(Random Forest)"];
        LR [label="Baseline Model\\n(Logistic Regression)"];
        
        Eval [label="Cross-Validation &\\nMetric Benchmarking"];
    }

    subgraph cluster_output {
        label = "Artifacts";
        style = "dashed";
        color = "#ced4da";
        fontname="Inter";
        
        PipelinePKL [label="Serialized Pipeline\\n(pipeline.pkl)", shape=note, fillcolor="#fff3cd", color="#856404"];
        MetricsJSON [label="Evaluation Metrics\\n(metrics.json)", shape=note];
    }

    RawData -> Load;
    Load -> Split;
    Split -> NumScale [label="Numerical Features"];
    Split -> CatEnc [label="Categorical Features"];
    
    NumScale -> SMOTE;
    CatEnc -> SMOTE;
    
    SMOTE -> XGB;
    SMOTE -> RF;
    SMOTE -> LR;
    
    XGB -> Eval;
    RF -> Eval;
    LR -> Eval;

    Eval -> PipelinePKL [label="Best Model Selection"];
    Eval -> MetricsJSON;
}
"""
    dot_file = REPORTS_DIR / "pipeline_diagram.dot"
    png_file = REPORTS_DIR / "pipeline_diagram.png"
    
    with open(dot_file, "w") as f:
        f.write(dot_content)
        
    try:
        subprocess.run(["dot", "-Tpng", str(dot_file), "-o", str(png_file)], check=True)
        logger.info(f"Pipeline diagram successfully generated at {png_file}")
    except FileNotFoundError:
        logger.error("Graphviz 'dot' command not found. Ensure Graphviz is installed via Homebrew.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate pipeline diagram: {e}")
    finally:
        if dot_file.exists():
            dot_file.unlink()

def generate_architecture_diagram():
    dot_content = """digraph Architecture {
    rankdir=TB;
    node [shape=box, style="rounded,filled", fontname="Inter", fillcolor="#f8f9fa", color="#343a40", penwidth=1.5];
    edge [color="#495057", penwidth=1.2, fontname="Inter", fontsize=10];

    subgraph cluster_client {
        label = "Presentation Layer (Client)";
        style = "filled";
        fillcolor = "#f1f3f5";
        color = "#dee2e6";
        fontname="Inter";
        
        UI [label="Streamlit Web Dashboard\\n(app.py)", shape=rect, fillcolor="#cff4fc", color="#055160"];
    }

    subgraph cluster_server {
        label = "Application Layer (Server)";
        style = "filled";
        fillcolor = "#f1f3f5";
        color = "#dee2e6";
        fontname="Inter";
        
        InferenceEngine [label="Inference Engine\\n(Prediction Service)", fillcolor="#d1e7dd"];
        Analytics [label="Analytics & Metrics Loader"];
        Interpretability [label="SHAP Explainer Module"];
    }

    subgraph cluster_storage {
        label = "Data & Artifact Storage";
        style = "filled";
        fillcolor = "#f1f3f5";
        color = "#dee2e6";
        fontname="Inter";
        
        ModelStore [label="Model Registry\\n(pipeline.pkl)", shape=cylinder, fillcolor="#e2e3e5"];
        MetricsStore [label="Metrics & Reports\\n(metrics.json / CSV)", shape=folder, fillcolor="#e2e3e5"];
    }

    UI -> InferenceEngine [label="Session Features (JSON/Dict)"];
    InferenceEngine -> UI [label="Conversion Probability & Routing Tier"];
    
    UI -> Analytics [label="Request Dashboard Data"];
    Analytics -> MetricsStore [label="Read"];
    Analytics -> UI [label="Render KPIs & Charts"];
    
    InferenceEngine -> ModelStore [label="Load Pipeline"];
    Interpretability -> ModelStore [label="Load Weights"];
}
"""
    dot_file = REPORTS_DIR / "architecture_diagram.dot"
    png_file = REPORTS_DIR / "architecture_diagram.png"
    
    with open(dot_file, "w") as f:
        f.write(dot_content)
        
    try:
        subprocess.run(["dot", "-Tpng", str(dot_file), "-o", str(png_file)], check=True)
        logger.info(f"Architecture diagram successfully generated at {png_file}")
    except FileNotFoundError:
        logger.error("Graphviz 'dot' command not found. Ensure Graphviz is installed via Homebrew.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate architecture diagram: {e}")
    finally:
        if dot_file.exists():
            dot_file.unlink()

if __name__ == "__main__":
    logger.info("Generating system architecture and pipeline diagrams...")
    generate_pipeline_diagram()
    generate_architecture_diagram()
