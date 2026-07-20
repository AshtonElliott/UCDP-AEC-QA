import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines

def _ensure_assets_dir():
    os.makedirs("assets", exist_ok=True)

def clear_assets_dir():
    """Wipes all existing PNG files in the assets folder to prevent clutter."""
    _ensure_assets_dir()
    old_files = glob.glob(os.path.join("assets", "*.png"))
    for file in old_files:
        try:
            os.remove(file)
        except Exception:
            pass

def generate_performance_quadrant(df, model_col, filename="master_quadrant.png"):
    _ensure_assets_dir()
    plt.figure(figsize=(11, 8)) 
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        # plot strict (circle)
        plt.scatter(row['Lexical (EM)'], row['Contextual F1'], 
                    color=color, marker='o', s=250, alpha=0.8)
        
        # plot eelaxed (triangle)
        plt.scatter(row['Token IoU'], row['Deduped F1'], 
                    color=color, marker='^', s=250, alpha=0.8)
        
        # draw dotted line connecting them
        plt.plot([row['Lexical (EM)'], row['Token IoU']], 
                 [row['Contextual F1'], row['Deduped F1']], 
                 color=color, linestyle=':', alpha=0.6)

    plt.axvline(x=df['Lexical (EM)'].mean(), color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=df['Contextual F1'].mean(), color='gray', linestyle='--', alpha=0.3)

    plt.title("Model Performance Shift: Strict vs. Relaxed Evaluation", fontsize=14, pad=15)
    plt.xlabel("Lexical Match (SQuAD EM  ➔  Token IoU)")
    plt.ylabel("Semantic Match (DeBERTa F1  ➔  Deduped F1)")
    plt.grid(True, alpha=0.3)
    
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='s', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    first_legend = plt.legend(handles=model_handles, title="Models", 
                              bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().add_artist(first_legend) 

    strict_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Strict (EM / F1)')
    relaxed_marker = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Relaxed (IoU / Dedup F1)')
    
    plt.legend(handles=[strict_marker, relaxed_marker], title="Metric Type", 
               bbox_to_anchor=(1.05, 0.65), loc='upper left')

    plt.tight_layout()
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight') 
    plt.close()

def generate_verbosity_scatter(df, model_col, filename="verbosity_vs_accuracy.png"):
    _ensure_assets_dir()
    plt.figure(figsize=(9, 6))
    
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        # X = Avg Spans, Y = Deduped F1
        plt.scatter(row['Avg Spans'], row['Deduped F1'], 
                    color=color, marker='o', s=300, alpha=0.9, edgecolor='white')

    plt.title("Verbosity vs. Semantic Accuracy", fontsize=14, pad=15)
    plt.xlabel("Average Spans Generated per Article", fontsize=12)
    plt.ylabel("Relaxed Semantic Score (Deduped F1)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # clean external legend for the models
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='o', 
                                   linestyle='None', markersize=10, label=m.replace('*', '')) for m in models]
    
    handles_to_plot = model_handles
    
    if len(df) > 1 and df['Avg Spans'].nunique() > 1:
        try:
            z = np.polyfit(df['Avg Spans'], df['Deduped F1'], 2)
            p = np.poly1d(z)
            x_seq = np.linspace(df['Avg Spans'].min() * 0.9, df['Avg Spans'].max() * 1.1, 100)
            plt.plot(x_seq, p(x_seq), color='red', linestyle='--', alpha=0.4)
            
            trend_line = mlines.Line2D([], [], color='red', linestyle='--', alpha=0.4, label='Polynomial Trendline')
            handles_to_plot.append(trend_line)
        except Exception:
            pass

    plt.legend(handles=handles_to_plot, title="Models & Trends", 
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def generate_task_heatmap(df, filename="task_complexity_heatmap.png"):
    _ensure_assets_dir()
    plt.figure(figsize=(10, 6))
    
    # pivot the data so models are rows and questions are columns
    heatmap_data = df.pivot(index="Model Target", columns="Question Track", values="Dedup F1")
    
    # sort the heatmap descending based on Question 1 performance
    if "Question 1" in heatmap_data.columns:
        heatmap_data = heatmap_data.sort_values(by="Question 1", ascending=False)
    
    mask = heatmap_data.isnull()
    
    sns.heatmap(heatmap_data, mask=mask, annot=True, cmap="YlGnBu", fmt=".4f", cbar_kws={'label': 'Relaxed Semantic Score (Dedup F1)'})
    
    plt.title("Task Complexity Degradation: Q1 vs Q2", fontsize=14, pad=15)
    plt.ylabel("")
    plt.xlabel("")
    plt.tight_layout()
    
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()