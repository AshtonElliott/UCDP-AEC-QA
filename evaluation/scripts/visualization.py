import os
import matplotlib.pyplot as plt
import seaborn as sns

def _ensure_assets_dir():
    os.makedirs("assets", exist_ok=True)

def generate_rank_distribution_plot(df, x_col, y_col, title, x_label, y_label, filename):
    _ensure_assets_dir()
    plt.figure(figsize=(9, 6))
    
    # boxplot shows the distribution (median, quartiles) for each discrete human score bucket
    ax = sns.boxplot(x=x_col, y=y_col, data=df, color="whitesmoke", showfliers=False)
    
    # stripplot adds the individual data points with jitter so they don't overlap vertically
    sns.stripplot(x=x_col, y=y_col, data=df, alpha=0.6, size=6, palette="dark:blue", jitter=True)
    
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.ylim(-0.05, 1.05)
    
    labels = [item.get_text() for item in ax.get_xticklabels()]
    clean_labels = [f"{float(label):.2f}" if label.replace('.','',1).isdigit() else label for label in labels]
    ax.set_xticklabels(clean_labels, rotation=45)
    
    plt.tight_layout()
    
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def generate_leaderboard_bar(df, model_col, metrics, filename):
    _ensure_assets_dir()
    
    # melt the dataframe for side-by-side plotting
    df_melted = df.melt(id_vars=[model_col], value_vars=metrics, 
                        var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, y=model_col, x="Score", hue="Metric", palette="viridis")
    
    plt.title("LLM Performance Comparison")
    plt.xlabel("Score (0.0 to 1.0)")
    plt.ylabel("")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def generate_performance_quadrant(df, model_col, x_col, y_col, filename):
    _ensure_assets_dir()
    plt.figure(figsize=(8, 8))
    
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=model_col, 
                    s=200, palette="deep", legend=False)
    
    # add text labels for each model
    for i in range(df.shape[0]):
        plt.text(df[x_col][i] + 0.005, 
                 df[y_col][i] + 0.005, 
                 df[model_col][i], 
                 horizontalalignment='left', 
                 size='medium', color='black', weight='semibold')

    # draw median lines to create quadrants
    plt.axvline(x=df[x_col].mean(), color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=df[y_col].mean(), color='gray', linestyle='--', alpha=0.5)

    plt.title("Model Performance: Strict Extraction vs. Semantic Understanding")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join("assets", filename)
    plt.savefig(filepath, dpi=300)
    plt.close()