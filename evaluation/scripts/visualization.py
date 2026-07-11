import os
import sys
import glob
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
        except Exception as e:
            pass

def generate_leaderboard_bar(df, model_col, filename):
    _ensure_assets_dir()
    
    # We only plot the 4 machine metrics, ignoring the human score for the bar chart
    metrics = ["Avg_SQuAD_EM", "Avg_IoU", "Avg_BS_F1", "Avg_BS_F1_Dedup"]
    metric_labels = {
        "Avg_SQuAD_EM": "Strict Lexical (SQuAD EM)",
        "Avg_IoU": "Relaxed Lexical (Token IoU)",
        "Avg_BS_F1": "Strict Semantic (DeBERTa F1)",
        "Avg_BS_F1_Dedup": "Relaxed Semantic (Dedup F1)"
    }
    
    df_melted = df.melt(id_vars=[model_col], value_vars=metrics, 
                        var_name="Metric", value_name="Score")
    df_melted["Metric"] = df_melted["Metric"].map(metric_labels)

    plt.figure(figsize=(11, 6))
    
    # Using a clean color palette to differentiate Lexical (Blues) vs Semantic (Greens)
    sns.barplot(data=df_melted, y=model_col, x="Score", hue="Metric", palette="Paired")
    
    plt.title("Model Capture Profiles: Lexical (Text) vs. Semantic (Meaning)")
    plt.xlabel("Score (0.0 to 1.0)")
    plt.ylabel("")
    plt.xlim(0, 1.0)
    
    # Move legend outside the plot so it doesn't overlap bars
    plt.legend(title="Evaluation Track", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def generate_performance_quadrant(df, model_col, filename):
    _ensure_assets_dir()
    # Increased width slightly to make room for the external legends
    plt.figure(figsize=(11, 8)) 
    
    # Create a color palette for the models
    models = df[model_col].unique()
    palette = sns.color_palette("deep", len(models))
    color_map = dict(zip(models, palette))

    for _, row in df.iterrows():
        model = row[model_col]
        color = color_map[model]
        
        # Plot Strict (Circle)
        plt.scatter(row['Avg_SQuAD_EM'], row['Avg_BS_F1'], 
                    color=color, marker='o', s=250, alpha=0.8)
        
        # Plot Relaxed (Triangle)
        plt.scatter(row['Avg_IoU'], row['Avg_BS_F1_Dedup'], 
                    color=color, marker='^', s=250, alpha=0.8)
        
        # Draw a dotted line connecting them to show the "shift"
        plt.plot([row['Avg_SQuAD_EM'], row['Avg_IoU']], 
                 [row['Avg_BS_F1'], row['Avg_BS_F1_Dedup']], 
                 color=color, linestyle=':', alpha=0.6)
        
        # NOTE: Removed plt.text() to eliminate chart clutter

    # Draw median lines based on strict scores to anchor the visual
    plt.axvline(x=df['Avg_SQuAD_EM'].mean(), color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=df['Avg_BS_F1'].mean(), color='gray', linestyle='--', alpha=0.3)

    plt.title("Model Performance Shift: Strict vs. Relaxed Evaluation")
    plt.xlabel("Lexical Match (SQuAD EM  ➔  Token IoU)")
    plt.ylabel("Semantic Match (DeBERTa F1  ➔  Deduped F1)")
    plt.grid(True, alpha=0.3)
    
    # --- Dual Legend System ---
    
    # 1. Models Legend (Colors)
    model_handles = [mlines.Line2D([], [], color=color_map[m], marker='s', 
                                   linestyle='None', markersize=10, label=m) for m in models]
    first_legend = plt.legend(handles=model_handles, title="Models", 
                              bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Matplotlib drops the first legend when you add a second one, so we must add it back manually
    plt.gca().add_artist(first_legend) 

    # 2. Metric Type Legend (Shapes)
    strict_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Strict (EM / F1)')
    relaxed_marker = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Relaxed (IoU / Dedup F1)')
    
    plt.legend(handles=[strict_marker, relaxed_marker], title="Metric Type", 
               bbox_to_anchor=(1.05, 0.65), loc='upper left')

    plt.tight_layout()
    filepath = os.path.join("assets", filename)
    # Added bbox_inches='tight' to ensure the external legends don't get cropped
    plt.savefig(filepath, dpi=300, bbox_inches='tight') 
    plt.close()

def generate_scaling_plot(data_by_question, filename="pipeline_scaling.png"):
    """
    Plots a multi-line linear cost model comparing the compute time 
    scaling factors between different evaluation questions.
    """
    _ensure_assets_dir()
    plt.figure(figsize=(10, 6))
    
    # Use a clean, distinct color palette for the lines
    colors = sns.color_palette("Set1", len(data_by_question))
    
    max_x = 10
    max_y = 1.0
    
    for i, (q_num, metrics) in enumerate(sorted(data_by_question.items())):
        sizes = metrics["sizes"]
        times = metrics["times"]
        
        max_x = max(max_x, max(sizes))
        max_y = max(max_y, max(times))
        
        plt.plot(sizes, times, marker='o', color=colors[i], linewidth=2.5, 
                 markersize=6, label=f'Question {q_num}')
        
    plt.title("Evaluation Pipeline Cost Model: Question Complexity Comparison", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Dataset Size (Number of News Articles)", fontsize=11)
    plt.ylabel("Compute Time Elapsed (Seconds)", fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, max_x * 1.05)
    plt.ylim(0, max_y * 1.1)
    plt.legend(title="Evaluation Track", loc="upper left")
    plt.tight_layout()
    
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Multi-line scaling chart saved successfully to assets/{filename}", file=sys.stderr)