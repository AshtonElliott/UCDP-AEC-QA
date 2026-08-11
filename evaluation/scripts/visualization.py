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
    """Plots Answerability (NoAns Acc) vs. Extraction Quality (HasAns Dedup BERT)."""
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    x_mid = df['NoAns Acc'].median() if not df['NoAns Acc'].isnull().all() else 0.5
    y_mid = df['HasAns Dedup BERT'].median() if not df['HasAns Dedup BERT'].isnull().all() else 0.5

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        ax.scatter(row['NoAns Acc'], row['HasAns Dedup BERT'], 
                   color=color, marker='o', s=300, alpha=0.9, edgecolor='white')

    ax.axvline(x=x_mid, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=y_mid, color='gray', linestyle='--', alpha=0.5)

    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    
    if len(df) > 0:
        ax.text(df['NoAns Acc'].max(), df['HasAns Dedup BERT'].max(), 'Ideal Performers\n(Accurate & Safe)', 
                fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=bbox_props, alpha=0.6)
        ax.text(df['NoAns Acc'].min(), df['HasAns Dedup BERT'].max(), 'Hallucinators\n(Talkative & Unsafe)', 
                fontsize=10, verticalalignment='top', horizontalalignment='left', bbox=bbox_props, alpha=0.6)

    ax.set_title("SQuAD 2.0 Behavior: Abstention vs. Extraction Quality", fontsize=14, pad=15)
    ax.set_xlabel("Gatekeeping / Abstention (NoAns)", fontsize=12)
    ax.set_ylabel("Extraction Quality (HasAns Deduped BERTScore)", fontsize=12)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=':', alpha=0.4)
   
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='o', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    ax.legend(handles=model_handles, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight') 
    plt.close(fig)

def generate_strict_vs_relaxed_quadrant(df, model_col="Model", filename="strict_vs_relaxed.png"):
    """Plots the performance shift from Strict (Span F1) to Relaxed (Token F1)."""
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(11, 8))
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        ax.scatter(row['HasAns Span F1'], row['HasAns BERTScore'], color=color, marker='o', s=250, alpha=0.8)
        ax.scatter(row['HasAns Token F1'], row['HasAns Dedup BERT'], color=color, marker='^', s=250, alpha=0.8)
        
        ax.plot([row['HasAns Span F1'], row['HasAns Token F1']], 
                [row['HasAns BERTScore'], row['HasAns Dedup BERT']], 
                color=color, linestyle=':', alpha=0.6)

    if len(df) > 0:
        ax.axvline(x=df['HasAns Span F1'].mean(), color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=df['HasAns BERTScore'].mean(), color='gray', linestyle='--', alpha=0.3)

    ax.set_title("Model Performance Shift: Strict vs. Relaxed Evaluation (HasAns)", fontsize=14, pad=15)
    ax.set_xlabel("Text Match (Span F1  ➔  SQuAD Token F1)", fontsize=12)
    ax.set_ylabel("Semantic Match (BERTScore  ➔  Deduped BERTScore)", fontsize=12)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
   
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='s', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    first_legend = ax.legend(handles=model_handles, title="Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(first_legend) 

    strict_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Strict (Span F1 / BERTScore)')
    relaxed_marker = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Relaxed (Token F1 / Dedup BERT)')
    
    ax.legend(handles=[strict_marker, relaxed_marker], title="Metric Type", bbox_to_anchor=(1.05, 0.65), loc='upper left')

    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight') 
    plt.close(fig)

def generate_verbosity_scatter(df, model_col="Model", filename="verbosity_vs_accuracy.png"):
    """Plots Avg Spans vs HasAns Dedup BERT."""
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
    
    ax.set_ylim(-0.05, 1.05)
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
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = df.pivot(index="Model Target", columns="Question Track", values="Overall Dedup BERT")
    if "Question 1" in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values(by="Question 1", ascending=False)
    
    sns.heatmap(heatmap_data, mask=heatmap_data.isnull(), annot=True, cmap="flare", fmt=".4f", 
                linewidths=0, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Deduped BERTScore (Overall)'}, ax=ax)
    
    ax.set_title("Task Complexity Degradation (Overall Score)", fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_task_heatmap_hasans(df, filename="task_complexity_heatmap_hasans.png"):
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    heatmap_data = df.pivot(index="Model Target", columns="Question Track", values="HasAns Dedup BERT")
    if "Question 1" in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values(by="Question 1", ascending=False)
    
    sns.heatmap(heatmap_data, mask=heatmap_data.isnull(), annot=True, cmap="flare", fmt=".4f", 
                linewidths=0, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Deduped BERTScore (HasAns)'}, ax=ax)
    
    ax.set_title("True Extraction Complexity (HasAns Only)", fontsize=14, pad=15)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_classification_dropoff(df, filename="classification_dropoff_q2.png"):
    _ensure_assets_dir()
    
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
    
    rects1 = ax.bar(x - width/2, text_scores, width, label='Extracted Correct Text (Span F1)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, label_scores, width, label='Assigned Correct Label (Labeled Span F1)', color='#dd8452')

    ax.set_ylabel('Score (HasAns)', fontsize=12)
    ax.set_ylim(0, 1.05) 
    ax.set_title('Question 2: Reading Comprehension vs. Classification Reasoning', fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m).replace('*', '') for m in models], rotation=45, ha='right')
    
    # Legend tucked inside the graph
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    filepath = ASSETS_DIR / filename
    plt.savefig(str(filepath), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_pr_scatter(df, x_col, y_col, title, filename):
    """
    Plot A: Paired Bar Chart (Precision vs. Recall) to clearly show extraction imbalances.
    """
    _ensure_assets_dir()
    
    df = df.sort_values(by=x_col, ascending=False)
    
    models = df["Model"].tolist()
    recall_scores = df[x_col].tolist()
    precision_scores = df[y_col].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, recall_scores, width, label='Recall (Found the correct text)', color='#4c72b0')
    rects2 = ax.bar(x + width/2, precision_scores, width, label='Precision (Only extracted the right text)', color='#dd8452')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0, 1.05) 
    ax.set_title(title.replace("scatter", "Bar Chart").replace("Scatter", "Bar Chart"), fontsize=14, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(m).replace('*', '') for m in models], rotation=45, ha='right')
    
    # Legend tucked inside the graph
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_category_pr_grid(df, filename="q2_category_pr_grid.png"):
    """
    Plot B: Grid panels = Models. X/Y = Recall/Precision. Colors = Event Categories.
    (No F1 line to prevent visual clutter).
    """
    _ensure_assets_dir()
    
    cat_data = []
    for _, row in df.iterrows():
        model = row.get('model') or row.get('Model')
        stats = row.get('cat_stats', {})
        if not isinstance(stats, dict):
            continue
        for cat, counts in stats.items():
            cat_data.append({
                "Model": str(model).replace('*', ''), 
                "Category": str(cat).title(),
                "TP": counts.get("TP", 0),
                "FP": counts.get("FP", 0),
                "FN": counts.get("FN", 0)
            })
            
    if not cat_data:
        return
        
    cat_df = pd.DataFrame(cat_data)
    agg_df = cat_df.groupby(["Model", "Category"]).sum().reset_index()
    
    agg_df['Precision'] = agg_df.apply(lambda r: r['TP'] / (r['TP'] + r['FP']) if (r['TP'] + r['FP']) > 0 else 0.0, axis=1)
    agg_df['Recall'] = agg_df.apply(lambda r: r['TP'] / (r['TP'] + r['FN']) if (r['TP'] + r['FN']) > 0 else 0.0, axis=1)
    
    support_df = agg_df.groupby("Category").apply(lambda x: x['TP'].sum() + x['FN'].sum())
    top_cats = support_df.nlargest(8).index.tolist()
    plot_df = agg_df[agg_df['Category'].isin(top_cats)]
    
    if plot_df.empty: return

    palette = sns.color_palette("deep", len(top_cats))
    
    g = sns.FacetGrid(plot_df, col="Model", col_wrap=3, height=3.5, aspect=1.1, hue="Category", palette=palette)
    
    g.map_dataframe(sns.scatterplot, x="Recall", y="Precision", s=150, alpha=0.9, edgecolor='white')
    
    for ax in g.axes.flat:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.5)
    
    g.set_axis_labels("Recall", "Precision")
    g.set_titles(col_template="{col_name}")
    
    g.add_legend(title="Event Categories", bbox_to_anchor=(1.02, 0.5), loc='center left')
    
    g.fig.subplots_adjust(top=0.88, hspace=0.4) 
    g.fig.suptitle("Question 2: Category Precision vs. Recall by Model", fontsize=16)
    
    g.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close()