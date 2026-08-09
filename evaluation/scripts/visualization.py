import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

def _ensure_assets_dir():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def clear_assets_dir():
    """Wipes all existing PNG files in the assets folder to prevent clutter"""
    _ensure_assets_dir()
    for file in ASSETS_DIR.glob("*.png"):
        try:
            file.unlink(missing_ok=True)
        except Exception:
            pass

def generate_performance_quadrant(df, model_col="Model", filename="master_quadrant.png"):
    """
    Plots Answerability (NoAns Acc) vs. Extraction Quality (HasAns Dedup BERT).
    Creates 4 quadrants showing model 'personalities'.
    """
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    # Calculate medians to draw quadrant crosshairs safely
    x_mid = df['NoAns Acc'].median() if not df['NoAns Acc'].isnull().all() else 0.5
    y_mid = df['HasAns Dedup BERT'].median() if not df['HasAns Dedup BERT'].isnull().all() else 0.5

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        ax.scatter(row['NoAns Acc'], row['HasAns Dedup BERT'], 
                   color=color, marker='o', s=300, alpha=0.9, edgecolor='white')

    # Draw Quadrant lines
    ax.axvline(x=x_mid, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=y_mid, color='gray', linestyle='--', alpha=0.5)

    # Add quadrant labels
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    
    # Only draw text if we have valid ranges to avoid plotting errors
    if len(df) > 0:
        ax.text(df['NoAns Acc'].max(), df['HasAns Dedup BERT'].max(), 'Ideal Performers\n(Accurate & Safe)', 
                fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=bbox_props, alpha=0.6)
        ax.text(df['NoAns Acc'].min(), df['HasAns Dedup BERT'].max(), 'Hallucinators\n(Talkative & Unsafe)', 
                fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=bbox_props, alpha=0.6)

    ax.set_title("SQuAD 2.0 Behavior: Abstention vs. Extraction Quality", fontsize=14, pad=15)
    ax.set_xlabel("Gatekeeping / Abstention (NoAns)", fontsize=12)
    ax.set_ylabel("Extraction Quality (HasAns Deduped BERTScore)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.4)
   
    # Legend
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='o', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    ax.legend(handles=model_handles, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight') 
    plt.close(fig)


def generate_strict_vs_relaxed_quadrant(df, model_col="Model", filename="strict_vs_relaxed.png"):
    """
    Plots the performance shift from Strict (Span F1 / BERTScore) to Relaxed (Token F1 / Dedup BERT).
    Uses HasAns metrics to isolate pure text extraction mechanics.
    """
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(11, 8))
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        # Plot Strict (Circle)
        ax.scatter(row['HasAns Span F1'], row['HasAns BERTScore'], 
                   color=color, marker='o', s=250, alpha=0.8)
        
        # Plot Relaxed (Triangle)
        ax.scatter(row['HasAns Token F1'], row['HasAns Dedup BERT'], 
                   color=color, marker='^', s=250, alpha=0.8)
        
        # Draw dotted line connecting them
        ax.plot([row['HasAns Span F1'], row['HasAns Token F1']], 
                [row['HasAns BERTScore'], row['HasAns Dedup BERT']], 
                color=color, linestyle=':', alpha=0.6)

    # Calculate medians for safe crosshairs
    if len(df) > 0:
        ax.axvline(x=df['HasAns Span F1'].mean(), color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=df['HasAns BERTScore'].mean(), color='gray', linestyle='--', alpha=0.3)

    ax.set_title("Model Performance Shift: Strict vs. Relaxed Evaluation (HasAns)", fontsize=14, pad=15)
    ax.set_xlabel("Text Match (Span F1  ➔  SQuAD Token F1)", fontsize=12)
    ax.set_ylabel("Semantic Match (BERTScore  ➔  Deduped BERTScore)", fontsize=12)
    ax.grid(True, alpha=0.3)
   
    # Models Legend (Colors)
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='s', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    first_legend = ax.legend(handles=model_handles, title="Models", 
                             bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(first_legend) 

    # Metric Type Legend (Shapes)
    strict_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Strict (Span F1 / BERTScore)')
    relaxed_marker = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Relaxed (Token F1 / Dedup BERT)')
    
    ax.legend(handles=[strict_marker, relaxed_marker], title="Metric Type", 
              bbox_to_anchor=(1.05, 0.65), loc='upper left')

    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight') 
    plt.close(fig)


def generate_verbosity_scatter(df, model_col="Model", filename="verbosity_vs_accuracy.png"):
    """
    Plots Avg Spans vs HasAns Dedup BERT to see if verbosity inflates semantic scores.
    """
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(9, 6))
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        ax.scatter(row['Avg Spans'], row['HasAns Dedup BERT'], 
                   color=color, marker='o', s=300, alpha=0.9, edgecolor='white')

    ax.set_title("Verbosity vs. Semantic Accuracy", fontsize=14, pad=15)
    ax.set_xlabel("Average Spans Generated per Article", fontsize=12)
    ax.set_ylabel("Extraction Quality (HasAns Deduped BERTScore)", fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    handles_to_plot = [mlines.Line2D([], [], color=color_map[m], marker='o', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    
    if len(df) > 1 and df['Avg Spans'].nunique() > 1:
        try:
            z = np.polyfit(df['Avg Spans'], df['HasAns Dedup BERT'], 2)
            p = np.poly1d(z)
            x_seq = np.linspace(df['Avg Spans'].min() * 0.9, df['Avg Spans'].max() * 1.1, 100)
            ax.plot(x_seq, p(x_seq), color='red', linestyle='--', alpha=0.4)
            trend_line = mlines.Line2D([], [], color='red', linestyle='--', alpha=0.4, label='Polynomial Trendline')
            handles_to_plot.append(trend_line)
        except Exception:
            pass

    ax.legend(handles=handles_to_plot, title="Models & Trends", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_task_heatmap_overall(df, filename="task_complexity_heatmap_overall.png"):
    """Shows performance degradation using Overall Deduped BERTScore."""
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = df.pivot(index="Model Target", columns="Question Track", values="Overall Dedup BERT")
    if "Question 1" in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values(by="Question 1", ascending=False)
    
    sns.heatmap(heatmap_data, mask=heatmap_data.isnull(), annot=True, cmap="YlGnBu", fmt=".4f", 
                cbar_kws={'label': 'Deduped BERTScore (Overall)'}, ax=ax)
    
    ax.set_title("Task Complexity Degradation (Overall Score)", fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_task_heatmap_hasans(df, filename="task_complexity_heatmap_hasans.png"):
    """Shows true extraction capability by isolating HasAns Deduped BERTScore."""
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = df.pivot(index="Model Target", columns="Question Track", values="HasAns Dedup BERT")
    if "Question 1" in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values(by="Question 1", ascending=False)
    
    sns.heatmap(heatmap_data, mask=heatmap_data.isnull(), annot=True, cmap="OrRd", fmt=".4f", 
                cbar_kws={'label': 'Deduped BERTScore (HasAns)'}, ax=ax)
    
    ax.set_title("True Extraction Complexity (HasAns Only)", fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def generate_classification_dropoff(df, filename="classification_dropoff_q2.png"):
    """
    Creates a grouped bar chart for Question 2 showing the drop-off 
    between finding the text (Span F1) and classifying it correctly (Labeled Span F1).
    """
    _ensure_assets_dir()
    
    # Filter for Question 2 only and sort by Text Extraction capability
    if 'Question' not in df.columns: return
    df_q2 = df[df['Question'] == 2].copy()
    if df_q2.empty: return
    
    df_q2 = df_q2.sort_values(by='HasAns Span F1', ascending=False)
    
    models = df_q2['Model'].tolist()
    text_scores = df_q2['HasAns Span F1'].tolist()
    label_scores = df_q2['HasAns Labeled Span F1'].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw the bars
    rects1 = ax.bar(x - width/2, text_scores, width, label='Extracted Correct Text (Span F1)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, label_scores, width, label='Assigned Correct 8-Class Label (Labeled Span F1)', color='#dd8452')

    ax.set_ylabel('Score (HasAns)', fontsize=12)
    ax.set_title('Question 2: Reading Comprehension vs. Classification Reasoning', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)