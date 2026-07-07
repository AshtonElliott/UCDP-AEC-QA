import os
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
    plt.figure(figsize=(10, 8))
    
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
        
        # Text labels for the dots
        plt.text(row['Avg_SQuAD_EM'] + 0.005, row['Avg_BS_F1'] + 0.005, f"{model} (Strict)", size='small', color='black')
        plt.text(row['Avg_IoU'] + 0.005, row['Avg_BS_F1_Dedup'] + 0.005, f"{model} (Relaxed)", size='small', color='black', weight='bold')

    # Draw median lines based on strict scores to anchor the visual
    plt.axvline(x=df['Avg_SQuAD_EM'].mean(), color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=df['Avg_BS_F1'].mean(), color='gray', linestyle='--', alpha=0.3)

    plt.title("Model Performance Shift: Strict vs. Relaxed Evaluation")
    plt.xlabel("Lexical Match (SQuAD EM  ➔  Token IoU)")
    plt.ylabel("Semantic Match (DeBERTa F1  ➔  Deduped F1)")
    plt.grid(True, alpha=0.3)
    
    # Custom Legend for the shapes
    strict_marker = mlines.Line2D([], [], color='gray', marker='o', linestyle='None', markersize=10, label='Strict (EM / F1)')
    relaxed_marker = mlines.Line2D([], [], color='gray', marker='^', linestyle='None', markersize=10, label='Relaxed (IoU / Dedup F1)')
    plt.legend(handles=[strict_marker, relaxed_marker], loc='lower right', title="Metric Type")

    plt.tight_layout()
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()