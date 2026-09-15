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
    """Remove all existing PNG files in the assets directory."""
    _ensure_assets_dir()
    for file in ASSETS_DIR.glob("*.png"):
        try:
            file.unlink(missing_ok=True)
        except Exception:
            pass

def plot_prompt_engineering_impact(g_agg, filename="fig1_prompt_engineering_impact.png"):
    """
    Generate Figure 1: Dumbbell Plot.
    Configure X-Axis for HasAns BERTScore (0 to 1).
    Configure Y-Axis for Models.
    Compare Raw (Zero-Shot) versus Cookbook methods.
    """
    import traceback
    try:
        _ensure_assets_dir()
        
        # Filter data for non-thinking models (NT).
        df = g_agg[g_agg['Thinking_Raw'] == 'NT'].copy()
        
        # Pivot dataframe to align raw and cookbook values.
        pivot_df = df.pivot(index='Model_Raw', columns='Strategy_Raw', values='HasAns BERTScore')
        if 'raw' not in pivot_df.columns or 'cookbook' not in pivot_df.columns:
            return
            
        pivot_df = pivot_df.dropna(subset=['raw', 'cookbook']).sort_values(by='cookbook', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_range = range(len(pivot_df))
        ax.hlines(y=y_range, xmin=pivot_df['raw'], xmax=pivot_df['cookbook'], color='grey', alpha=0.4)
        
        ax.scatter(pivot_df['raw'], y_range, color='gray', alpha=0.8, label='Raw (Zero-Shot)', s=150)
        ax.scatter(pivot_df['cookbook'], y_range, color='blue', alpha=0.8, label='Cookbook (Prompted)', s=150)
        
        ax.set_yticks(y_range)
        ax.set_yticklabels(pivot_df.index)
        
        ax.set_xlim(0, 1)
        ax.set_xlabel('Extraction Quality (HasAns BERTScore)')
        ax.set_title('Impact of Prompt Engineering on Extraction Quality\n(Modes Plotted: Raw/No-Think vs. Cookbook/No-Think)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_prompt_engineering_impact]: {e}\n{traceback.format_exc()}")

def plot_hallucination_and_verbosity(g_agg, raw_df, filename="fig2_hallucination_and_verbosity.png"):
    """
    Generate Figure 2: Two-Panel Behavioral Plot.
    Configure Left Panel: Abstention (NoAns Acc) vs Extraction (HasAns BERTScore).
    Configure Right Panel: Avg Spans vs Extraction.
    """
    import traceback
    try:
        _ensure_assets_dir()
        
        # Filter data for Cookbook (No-Think) mode.
        g_df = g_agg[(g_agg['Strategy_Raw'] == 'cookbook') & (g_agg['Thinking_Raw'] == 'NT')].copy()
        
        # Calculate Average Spans from raw data.
        raw_filtered = raw_df[(raw_df['strategy'] == 'cookbook') & (raw_df['thinking'] == 'NT')]
        avg_spans = raw_filtered.groupby('model')['spans_generated'].mean().reset_index()
        avg_spans.rename(columns={'model': 'Model_Raw', 'spans_generated': 'Avg Spans'}, inplace=True)
        
        # Merge average spans into the global dataframe.
        plot_df = pd.merge(g_df, avg_spans, on='Model_Raw', how='inner')
        if plot_df.empty: return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
        
        palette = sns.color_palette("deep", len(plot_df))
        color_map = dict(zip(plot_df['Model_Raw'], palette))
        
        # Configure Left Panel.
        for _, row in plot_df.iterrows():
            ax1.scatter(row['NoAns Acc'], row['HasAns BERTScore'], color=color_map[row['Model_Raw']], s=250, alpha=0.8)
        
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel('Gatekeeping / Abstention Accuracy (NoAns Acc)')
        ax1.set_ylabel('Extraction Quality (HasAns BERTScore)')
        ax1.set_title('Extraction Accuracy vs. Abstention Safety')
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        ax1.text(0.95, 0.95, 'Ideal Performers\n(Accurate & Safe)', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
        ax1.text(0.05, 0.95, 'Hallucinators\n(Talkative & Unsafe)', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=bbox_props)

        # Configure Right Panel.
        for _, row in plot_df.iterrows():
            ax2.scatter(row['Avg Spans'], row['HasAns BERTScore'], color=color_map[row['Model_Raw']], s=250, alpha=0.8)
            
        if len(plot_df) > 1 and plot_df['Avg Spans'].nunique() > 1:
            try:
                z = np.polyfit(plot_df['Avg Spans'], plot_df['HasAns BERTScore'], 2)
                p = np.poly1d(z)
                x_seq = np.linspace(plot_df['Avg Spans'].min() * 0.9, plot_df['Avg Spans'].max() * 1.1, 100)
                ax2.plot(x_seq, p(x_seq), color='red', linestyle='--', alpha=0.4, label='Trendline')
            except:
                pass
                
        ax2.set_xlim(left=0) 
        ax2.set_xlabel('Average Extracted Spans per Article')
        ax2.set_title('Extraction Accuracy vs. Model Verbosity')
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        model_handles = [mlines.Line2D([], [], color=color_map[m], marker='o', 
                                       linestyle='None', markersize=10, label=m) for m in plot_df['Model_Raw']]
        ax2.legend(handles=model_handles, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        fig.suptitle('Global Reliability and Generation Dynamics\n(Modes Plotted: exclusively Cookbook/No-Think)', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_hallucination_and_verbosity]: {e}\n{traceback.format_exc()}")

def plot_classification_accuracy_drop(q_agg, filename="fig3_classification_accuracy_drop.png"):
    """
    Generate Figure 3: Hierarchical Reasoning Bottleneck.
    Isolate Question 2 only. Compare Span F1 vs Labeled Span F1.
    """
    import traceback
    try:
        _ensure_assets_dir()
        if 'Question' not in q_agg.columns: return
        df = q_agg[q_agg['Question'] == 2].copy()
        if df.empty: return
        
        # Group standard open-weight baselines on the left and thinking models on the right.
        # Identify thinking-capable models by checking for rows with Thinking_Raw == 'T'.
        thinking_models = df[df['Thinking_Raw'] == 'T']['Model_Raw'].unique()
        
        plot_rows = []
        
        # Process standard models.
        standard_models = [m for m in df['Model_Raw'].unique() if m not in thinking_models]
        for m in standard_models:
            # Retrieve the cookbook / NT state for standard models.
            row = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'cookbook') & (df['Thinking_Raw'] == 'NT')]
            if not row.empty:
                plot_rows.append({
                    'Label': f"{m}\n(No-Think)",
                    'Span F1': row.iloc[0]['HasAns Span F1'],
                    'Labeled Span F1': row.iloc[0]['HasAns Labeled Span F1']
                })
                
        # Process thinking models.
        for m in thinking_models:
            # Evaluate NT State.
            row_nt = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'cookbook') & (df['Thinking_Raw'] == 'NT')]
            if not row_nt.empty:
                plot_rows.append({
                    'Label': f"{m}\n(No-Think)",
                    'Span F1': row_nt.iloc[0]['HasAns Span F1'],
                    'Labeled Span F1': row_nt.iloc[0]['HasAns Labeled Span F1']
                })
            # Evaluate T State.
            row_t = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'cookbook') & (df['Thinking_Raw'] == 'T')]
            if not row_t.empty:
                plot_rows.append({
                    'Label': f"{m}\n(Think)",
                    'Span F1': row_t.iloc[0]['HasAns Span F1'],
                    'Labeled Span F1': row_t.iloc[0]['HasAns Labeled Span F1']
                })
                
        if not plot_rows: return
        plot_df = pd.DataFrame(plot_rows)
        
        labels = plot_df['Label']
        span_scores = plot_df['Span F1']
        labeled_span_scores = plot_df['Labeled Span F1']
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(x - width/2, span_scores, width, label='Text Localization (Span F1)', color='#4c72b0')
        ax.bar(x + width/2, labeled_span_scores, width, label='Hierarchical Categorization (Labeled Span F1)', color='#dd8452')
        
        ax.set_ylabel('Score (HasAns)')
        ax.set_ylim(0, 1.05)
        ax.set_title('Classification Dropoff: Reading Comprehension vs. Ontology Classification\n(Modes Plotted: Cookbook/NT for all, plus Cookbook/Think for thinking models)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_classification_accuracy_drop]: {e}\n{traceback.format_exc()}")

def plot_reasoning_vs_prompting_comparison(q_agg, filename="fig4_reasoning_vs_prompting.png"):
    """
    Generate Figure 4: Reasoning Token Ablation.
    Isolate the thinking-capable models.
    Facet plot by task complexity (Q1 vs Q2).
    Generate 4 bars per model: Raw/NT, Raw/T, Cookbook/NT, Cookbook/T.
    """
    import traceback
    try:
        _ensure_assets_dir()
        if 'Question' not in q_agg.columns: return
        
        thinking_models = q_agg[q_agg['Thinking_Raw'] == 'T']['Model_Raw'].unique()
        if len(thinking_models) == 0: return
        
        # Filter dataframe to include only thinking models.
        df = q_agg[q_agg['Model_Raw'].isin(thinking_models)].copy()
        
        plot_data = []
        for q in [1, 2]:
            q_df = df[df['Question'] == q]
            for m in thinking_models:
                m_df = q_df[q_df['Model_Raw'] == m]
                
                for strat, th in [('raw', 'NT'), ('raw', 'T'), ('cookbook', 'NT'), ('cookbook', 'T')]:
                    res = m_df[(m_df['Strategy_Raw'] == strat) & (m_df['Thinking_Raw'] == th)]
                    score = float(res.iloc[0]['HasAns BERTScore']) if not res.empty else 0.0
                    
                    state_name = f"{strat.title()} ({'Think' if th == 'T' else 'No-Think'})"
                    
                    plot_data.append({
                        'Question': f'Question {q}', 
                        'Model': m, 
                        'State': state_name, 
                        'Score': score
                    })
                
        plot_df = pd.DataFrame(plot_data)
        
        # Generate faceted plot using seaborn catplot.
        g = sns.catplot(
            data=plot_df, kind="bar",
            x="Model", y="Score", hue="State", col="Question",
            height=6, aspect=1.2, palette="muted"
        )
        
        g.set_axis_labels("", "Extraction Quality (HasAns BERTScore)")
        for ax in g.axes.flat:
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            
        g.fig.suptitle("Impact of Reasoning vs. Prompting on Task Complexity\n(Modes Plotted: All 4 modes (Raw/NT, Raw/T, Cookbook/NT, Cookbook/T))", y=1.05, fontsize=16)
        
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(g.fig)
    except Exception as e:
        print(f"\n[ERROR in plot_reasoning_vs_prompting_comparison]: {e}\n{traceback.format_exc()}")

# ----------------------------------------------------------------------------
# Provide requirements for Figure 5:
# Figure 5: Compute Latency vs. Extraction Quality (Time vs. Accuracy Trajectory)
# Configure Visual Structure: A 2D trajectory vector plot where each architecture is 
# represented by a directional arrow tracing its operational shift from 
# baseline to maximum effort.
# Configure Axes & Scaling:
# Map X-Axis to Wall-Clock Inference Latency (Average seconds per document).
# Map Y-Axis to Extraction Quality (HasAns BERTScore, 0.0 to 1.0).
# Select Plotted Data: Directional vectors originating at `Raw / No-Think` and 
# terminating at peak capability: `Cookbook / Think` for thinking models, 
# and `Cookbook / No-Think` for the others.
# Evaluate Analytical Takeaway: Quantifies computational return on investment. 
# Vertical vectors show high-efficiency accuracy gains; flat horizontal vectors 
# reveal wasted inference compute without improving quality.
# ----------------------------------------------------------------------------