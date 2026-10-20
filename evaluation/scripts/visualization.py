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

def create_placeholder_plot(filename, title, message):
    """Generate a placeholder plot indicating missing data."""
    _ensure_assets_dir()
    fig, ax = plt.subplots(figsize=(12, 7.0))
    ax.text(0.5, 0.5, f"Insufficient Data for Plot\n\n{message}", 
            horizontalalignment='center', verticalalignment='center', 
            fontsize=12, color='red', transform=ax.transAxes,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='red', boxstyle='round,pad=1'))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def _add_missing_data_text(ax, missing_details):
    if not missing_details: return
    msg = "Missing Data:\n" + "\n".join(missing_details[:10])
    if len(missing_details) > 10: msg += "\n... (truncated)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.05, 0.05, msg, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

def plot_prompt_engineering_impact(g_agg, filename="fig1_prompt_engineering_impact.png"):
    """Dumbbell plot comparing Raw vs. Cookbook for generative models."""
    import traceback
    try:
        _ensure_assets_dir()
        # Restrict strictly to generative models participating in prompt engineering (raw vs cookbook)
        df = g_agg[(g_agg['Thinking_Raw'] == 'NT') & (g_agg['Strategy_Raw'].isin(['raw', 'cookbook']))].copy()
        
        all_models = df['Model_Raw'].unique()
        missing_details = []
        for model in all_models:
            model_df = df[df['Model_Raw'] == model]
            strats = model_df['Strategy_Raw'].unique()
            if 'raw' not in strats: missing_details.append(f"{model}: missing Raw/NT")
            if 'cookbook' not in strats: missing_details.append(f"{model}: missing Cookbook/NT")
            
        pivot_df = df.pivot(index='Model_Raw', columns='Strategy_Raw', values='HasAns BERTScore')
        if 'raw' not in pivot_df.columns or 'cookbook' not in pivot_df.columns:
            msg = "Missing strategy data for NT models:\n" + "\n".join(missing_details[:10])
            if len(missing_details) > 10: msg += "\n... (truncated)"
            create_placeholder_plot(filename, 'Prompt Strategy Impact: Raw vs. Cookbook', msg)
            return
            
        pivot_df = pivot_df.dropna(subset=['raw', 'cookbook']).sort_values(by='cookbook', ascending=True)
        if pivot_df.empty:
            msg = "No models have both 'raw' and 'cookbook' data in NT mode.\n" + "\n".join(missing_details[:10])
            if len(missing_details) > 10: msg += "\n... (truncated)"
            create_placeholder_plot(filename, 'Prompt Strategy Impact: Raw vs. Cookbook', msg)
            return
        
        fig, ax = plt.subplots(figsize=(12, 7.0))
        y_range = range(len(pivot_df))
        ax.hlines(y=y_range, xmin=pivot_df['raw'], xmax=pivot_df['cookbook'], color='grey', alpha=0.4)
        
        ax.scatter(pivot_df['raw'], y_range, color='gray', alpha=0.8, label='Raw', s=150)
        ax.scatter(pivot_df['cookbook'], y_range, color='blue', alpha=0.8, label='Cookbook', s=150)
        
        ax.set_yticks(y_range)
        ax.set_yticklabels(pivot_df.index)
        
        ax.set_xlim(0, 1.0)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel('Extraction Quality (BERTScore F1)', fontsize=11)
        ax.set_ylabel('Model', fontsize=11)
        ax.set_title('Prompt Strategy Impact: Raw vs. Cookbook\n(Comparing baseline raw prompt vs. cookbook prompt on answerable texts)', fontsize=13, pad=12)
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='lower right')
        
        _add_missing_data_text(ax, missing_details)
        
        plt.tight_layout()
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_prompt_engineering_impact]: {e}\n{traceback.format_exc()}")

def plot_hallucination_and_verbosity(g_agg, raw_df, mode="all", filename="fig2_hallucination_and_verbosity.png"):
    """Extraction quality (bertscore_f1) vs Abstention accuracy (set_text_f1) and Verbosity (spans_generated)."""
    import traceback
    try:
        _ensure_assets_dir()
        
        if mode == "bert":
            g_df = g_agg[g_agg['Strategy_Raw'] == 'extractive'].copy()
            raw_filtered = raw_df[raw_df['strategy'] == 'extractive']
            subtitle_mode = "Extractive Models (BERT & ConfliBERT)"
        elif mode == "generative":
            g_df = g_agg[(g_agg['Strategy_Raw'] == 'cookbook') & (g_agg['Thinking_Raw'] == 'NT')].copy()
            raw_filtered = raw_df[(raw_df['strategy'] == 'cookbook') & (raw_df['thinking'] == 'NT')]
            subtitle_mode = "Generative LLMs (Cookbook/No-Think)"
        else: # "all"
            g_df = g_agg[((g_agg['Strategy_Raw'] == 'cookbook') & (g_agg['Thinking_Raw'] == 'NT')) | (g_agg['Strategy_Raw'] == 'extractive')].copy()
            raw_filtered = raw_df[((raw_df['strategy'] == 'cookbook') & (raw_df['thinking'] == 'NT')) | (raw_df['strategy'] == 'extractive')]
            subtitle_mode = "All Models"

        all_models = g_df['Model_Raw'].unique()
        missing_details = []
        
        avg_spans = raw_filtered.groupby('model')['spans_generated'].mean().reset_index()
        avg_spans.rename(columns={'model': 'Model_Raw', 'spans_generated': 'Avg Spans'}, inplace=True)
        
        plot_df = pd.merge(g_df, avg_spans, on='Model_Raw', how='inner')
        if plot_df.empty: 
            msg = f"No data found for {subtitle_mode}."
            create_placeholder_plot(filename, 'Model Reliability & Output Length', msg)
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.0), sharey=True)
        
        palette = sns.color_palette("deep", len(plot_df))
        color_map = dict(zip(plot_df['Model_Raw'], palette))
        
        for _, row in plot_df.iterrows():
            m_shape = '^' if row['Strategy_Raw'] == 'extractive' else 'o'
            ax1.scatter(row['NoAns Acc'], row['HasAns BERTScore'], color=color_map[row['Model_Raw']], marker=m_shape, s=250, alpha=0.85)
        
        ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlim(0, 1.0)
        ax1.set_ylim(0, 1.0)
        ax1.set_xticks(np.arange(0, 1.1, 0.2))
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        ax1.set_xlabel('Abstention Accuracy on Empty Articles (NoAns F1)', fontsize=11)
        ax1.set_ylabel('Extraction Quality (BERTScore F1)', fontsize=11)
        ax1.set_title('Quality vs. Abstention\n(Knowing when to say "No Answer")', fontsize=12, pad=10)
        ax1.grid(True, linestyle=':', alpha=0.6)
        
        bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
        ax1.text(0.98, 0.95, 'Reliable\n(Accurate & Safe)', transform=ax1.transAxes,
                fontsize=9.5, verticalalignment='top', horizontalalignment='right', bbox=bbox_props)
        ax1.text(0.02, 0.95, 'Hallucinates\n(Won\'t Abstain)', transform=ax1.transAxes,
                fontsize=9.5, verticalalignment='top', horizontalalignment='left', bbox=bbox_props)

        for _, row in plot_df.iterrows():
            m_shape = '^' if row['Strategy_Raw'] == 'extractive' else 'o'
            ax2.scatter(row['Avg Spans'], row['HasAns BERTScore'], color=color_map[row['Model_Raw']], marker=m_shape, s=250, alpha=0.85)
            
        if len(plot_df) > 1 and plot_df['Avg Spans'].nunique() > 1:
            try:
                z = np.polyfit(plot_df['Avg Spans'], plot_df['HasAns BERTScore'], 2)
                p = np.poly1d(z)
                x_seq = np.linspace(plot_df['Avg Spans'].min() * 0.9, plot_df['Avg Spans'].max() * 1.1, 100)
                ax2.plot(x_seq, p(x_seq), color='red', linestyle='--', alpha=0.4, label='Trendline')
            except:
                pass
                
        ax2.set_xlim(left=0) 
        ax2.set_xlabel('Average Spans Extracted per Article', fontsize=11)
        ax2.set_title('Quality vs. Output Length\n(Does extracting more text improve score?)', fontsize=12, pad=10)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        model_handles = []
        for _, row in plot_df.iterrows():
            m = row['Model_Raw']
            m_shape = '^' if row['Strategy_Raw'] == 'extractive' else 'o'
            lbl = f"{m} (Extractive)" if row['Strategy_Raw'] == 'extractive' else m
            model_handles.append(mlines.Line2D([], [], color=color_map[m], marker=m_shape, 
                                               linestyle='None', markersize=10, label=lbl))
            
        ax2.legend(handles=model_handles, title="Models", bbox_to_anchor=(1.02, 1), loc='upper left')
        _add_missing_data_text(ax2, missing_details)
        
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.86, wspace=0.18)
        fig.suptitle(f'Model Reliability and Output Length\n({subtitle_mode})', fontsize=14, y=0.97)
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_hallucination_and_verbosity]: {e}\n{traceback.format_exc()}")

def plot_classification_accuracy_drop(q_agg, mode="all", filename="fig3_classification_accuracy_drop.png"):
    """Classification dropoff: Text localization (set_text_f1) vs Hierarchical categorization (label_f1) on Q2."""
    import traceback
    try:
        _ensure_assets_dir()
        if 'Question' not in q_agg.columns: return
        df = q_agg[q_agg['Question'] == 2].copy()
        if df.empty: 
            create_placeholder_plot(filename, 'Question 2: Text Match vs. Category Match', "No data found for Question 2.")
            return
        
        plot_rows = []
        missing_details = []

        if mode in ["generative", "all"]:
            thinking_models = df[df['Thinking_Raw'] == 'T']['Model_Raw'].unique()
            standard_models = [m for m in df['Model_Raw'].unique() if m not in thinking_models and df[df['Model_Raw'] == m]['Strategy_Raw'].iloc[0] != 'extractive']
            
            for m in standard_models:
                row = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'cookbook') & (df['Thinking_Raw'] == 'NT')]
                if not row.empty:
                    plot_rows.append({
                        'Label': f"{m}\n(No-Think)",
                        'Span F1': row.iloc[0]['HasAns Span F1'],
                        'Labeled Span F1': row.iloc[0]['HasAns Labeled Span F1']
                    })
                else:
                    missing_details.append(f"{m}: missing Cookbook/NT")
                    
            for m in thinking_models:
                row_nt = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'cookbook') & (df['Thinking_Raw'] == 'NT')]
                if not row_nt.empty:
                    plot_rows.append({
                        'Label': f"{m}\n(No-Think)",
                        'Span F1': row_nt.iloc[0]['HasAns Span F1'],
                        'Labeled Span F1': row_nt.iloc[0]['HasAns Labeled Span F1']
                    })
                    
                row_t = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'cookbook') & (df['Thinking_Raw'] == 'T')]
                if not row_t.empty:
                    plot_rows.append({
                        'Label': f"{m}\n(Think)",
                        'Span F1': row_t.iloc[0]['HasAns Span F1'],
                        'Labeled Span F1': row_t.iloc[0]['HasAns Labeled Span F1']
                    })

        if mode in ["bert", "all"]:
            bert_models = [m for m in df['Model_Raw'].unique() if df[df['Model_Raw'] == m]['Strategy_Raw'].iloc[0] == 'extractive']
            for m in bert_models:
                row = df[(df['Model_Raw'] == m) & (df['Strategy_Raw'] == 'extractive')]
                if not row.empty:
                    lbl = f"{m}\n(Extractive)" if mode == "all" else m
                    plot_rows.append({
                        'Label': lbl,
                        'Span F1': row.iloc[0]['HasAns Span F1'],
                        'Labeled Span F1': row.iloc[0]['HasAns Labeled Span F1']
                    })

        if not plot_rows: 
            msg = "No valid models found matching the required modes."
            create_placeholder_plot(filename, 'Question 2: Text Match vs. Category Match', msg)
            return
            
        plot_df = pd.DataFrame(plot_rows)
        
        labels = plot_df['Label']
        span_scores = plot_df['Span F1']
        labeled_span_scores = plot_df['Labeled Span F1']
        
        x = np.arange(len(labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7.0))
        ax.bar(x - width/2, span_scores, width, label='Text Found (Span F1)', color='#4c72b0')
        ax.bar(x + width/2, labeled_span_scores, width, label='Text + Category Match (Labeled F1)', color='#dd8452')
        
        ax.set_ylabel('Score on Answerable Articles (F1)', fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_title('Question 2: Finding Text vs. Labeling Categories Correctly\n(Comparing exact text match against joint text and category match)', fontsize=13, pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.grid(True, axis='y', linestyle=':', alpha=0.6)
        
        _add_missing_data_text(ax, missing_details)
        
        plt.tight_layout()
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_classification_accuracy_drop]: {e}\n{traceback.format_exc()}")

def plot_reasoning_vs_prompting_comparison(q_agg, filename="fig4_reasoning_vs_prompting.png"):
    """Faceted bar chart comparing thinking tokens vs prompt engineering using bertscore_f1."""
    import traceback
    try:
        _ensure_assets_dir()
        if 'Question' not in q_agg.columns: return
        
        thinking_models = q_agg[q_agg['Thinking_Raw'] == 'T']['Model_Raw'].unique()
        if len(thinking_models) == 0: 
            create_placeholder_plot(filename, 'Thinking Models vs. Prompt Strategy (Raw vs. Cookbook)', 
                                    "No 'Thinking' (T) models found in the evaluation data.")
            return
        
        df = q_agg[q_agg['Model_Raw'].isin(thinking_models)].copy()
        
        plot_data = []
        missing_details = []
        for q in [1, 2]:
            q_df = df[df['Question'] == q]
            for m in thinking_models:
                m_df = q_df[q_df['Model_Raw'] == m]
                
                for strat, th in [('raw', 'NT'), ('raw', 'T'), ('cookbook', 'NT'), ('cookbook', 'T')]:
                    res = m_df[(m_df['Strategy_Raw'] == strat) & (m_df['Thinking_Raw'] == th)]
                    if res.empty:
                        missing_details.append(f"Q{q} {m}: missing {strat.title()}/{th}")
                        score = np.nan
                    else:
                        score = float(res.iloc[0]['HasAns BERTScore'])
                    
                    state_name = f"{'Cookbook' if strat == 'cookbook' else 'Raw'} ({'Thinking' if th == 'T' else 'No-Think'})"
                    
                    plot_data.append({
                        'Question': f'Question {q}', 
                        'Model': m, 
                        'State': state_name, 
                        'Score': score
                    })
                
        plot_df = pd.DataFrame(plot_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.0), sharey=True)
        
        q1_df = plot_df[plot_df['Question'] == 'Question 1']
        q2_df = plot_df[plot_df['Question'] == 'Question 2']
        
        state_order = [
            'Raw (No-Think)',
            'Raw (Thinking)',
            'Cookbook (No-Think)',
            'Cookbook (Thinking)'
        ]
        available_states = [s for s in state_order if s in plot_df['State'].unique()]
        
        sns.barplot(
            data=q1_df, x="Model", y="Score", hue="State", hue_order=available_states,
            ax=ax1, palette="muted"
        )
        sns.barplot(
            data=q2_df, x="Model", y="Score", hue="State", hue_order=available_states,
            ax=ax2, palette="muted"
        )
        
        ax1.set_title('Question 1: Weapons & Force Extraction', fontsize=12.5, pad=12, fontweight='bold')
        ax2.set_title('Question 2: Infrastructure Damage Extraction', fontsize=12.5, pad=12, fontweight='bold')
        
        for ax in (ax1, ax2):
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.set_xlabel('Model', fontsize=11)
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            
        ax1.set_ylabel('Extraction Quality (BERTScore F1)', fontsize=11)
        ax2.set_ylabel('')
        
        if ax1.get_legend() is not None:
            ax1.get_legend().remove()
        if ax2.get_legend() is not None:
            ax2.legend(title="Prompt & Reasoning State", loc='upper right', framealpha=0.9, fontsize=9.5)
            
        _add_missing_data_text(ax2, missing_details)
            
        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.18)
        fig.suptitle("Thinking Models vs. Prompt Strategy on Task Complexity\n(Comparing Raw vs. Cookbook across Standard and Thinking LLMs)", y=0.97, fontsize=14)
        
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_reasoning_vs_prompting_comparison]: {e}\n{traceback.format_exc()}")

def plot_bert_pretraining_impact(q_agg, filename="fig_b1_pretraining_impact.png"):
    """Compares BERT-base vs ConfliBERT-cont vs ConfliBERT-scr across Q1 and Q2 using set_text_f1 and bertscore_f1."""
    import traceback
    try:
        _ensure_assets_dir()
        if 'Question' not in q_agg.columns: return
        df = q_agg[q_agg['Strategy_Raw'] == 'extractive'].copy()
        if df.empty:
            create_placeholder_plot(filename, 'Figure B1: In-Domain Pretraining Impact', "No extractive BERT data found.")
            return

        def categorize_model(m):
            m_low = m.lower()
            if 'confli' in m_low and 'cont' in m_low: return 'ConfliBERT (Continual)'
            if 'confli' in m_low and 'scr' in m_low: return 'ConfliBERT (Scratch)'
            if 'bert' in m_low: return 'BERT-base (General)'
            return m

        df['Family'] = df['Model_Raw'].apply(categorize_model)
        
        family_q = df.groupby(['Family', 'Question'])[['HasAns Span F1', 'HasAns BERTScore']].mean().reset_index()
        family_q['Question'] = family_q['Question'].apply(lambda q: f"Q{int(q)}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.0), sharey=True)
        
        order = ['BERT-base (General)', 'ConfliBERT (Scratch)', 'ConfliBERT (Continual)']
        
        # Plot Question 1 (Weapons)
        q1_sub = family_q[family_q['Question'] == 'Q1'].set_index('Family').reindex(order).reset_index()
        x = np.arange(len(order))
        width = 0.35
        ax1.bar(x - width/2, q1_sub['HasAns Span F1'], width, label='Exact Text Match (Span F1)', color='#4c72b0')
        ax1.bar(x + width/2, q1_sub['HasAns BERTScore'], width, label='Semantic Match (BERTScore F1)', color='#55a868')
        ax1.set_title('Question 1: Weapons & Force Extraction', fontsize=12.5, pad=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(order, rotation=15, ha='right', fontsize=10)
        ax1.set_ylabel('Score on Answerable Articles (F1)', fontsize=11)
        ax1.set_ylim(0, 1.0)
        ax1.set_yticks(np.arange(0, 1.1, 0.2))
        ax1.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax1.legend(loc='upper left', framealpha=0.9)

        # Plot Question 2 (Infrastructure)
        q2_sub = family_q[family_q['Question'] == 'Q2'].set_index('Family').reindex(order).reset_index()
        ax2.bar(x - width/2, q2_sub['HasAns Span F1'], width, label='Exact Text Match (Span F1)', color='#4c72b0')
        ax2.bar(x + width/2, q2_sub['HasAns BERTScore'], width, label='Semantic Match (BERTScore F1)', color='#55a868')
        ax2.set_title('Question 2: Infrastructure Damage Extraction', fontsize=12.5, pad=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(order, rotation=15, ha='right', fontsize=10)
        ax2.set_yticks(np.arange(0, 1.1, 0.2))
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
        ax2.legend(loc='upper left', framealpha=0.9)

        fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.18)
        fig.suptitle('Pretraining Impact: General BERT vs. Conflict-Domain ConfliBERT\n(General BERT-base vs. Conflict-Trained ConfliBERT Continual & Scratch)', fontsize=14, y=0.97)
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_bert_pretraining_impact]: {e}\n{traceback.format_exc()}")

def plot_bert_casing_impact(q_agg, filename="fig_b2_casing_impact.png"):
    """Compares Cased vs Uncased for all BERT and ConfliBERT models across Q1 and Q2 using set_text_f1."""
    import traceback
    try:
        _ensure_assets_dir()
        if 'Question' not in q_agg.columns: return
        df = q_agg[q_agg['Strategy_Raw'] == 'extractive'].copy()
        if df.empty:
            create_placeholder_plot(filename, 'Case Sensitivity: Cased vs. Uncased Models', "No extractive BERT data found.")
            return

        def base_family(m):
            m_low = m.lower()
            if 'confli' in m_low and 'cont' in m_low: return 'ConfliBERT (Continual)'
            if 'confli' in m_low and 'scr' in m_low: return 'ConfliBERT (Scratch)'
            if 'bert' in m_low: return 'BERT-base'
            return m

        df['Family'] = df['Model_Raw'].apply(base_family)
        df['Casing'] = df['Model_Raw'].apply(lambda m: 'Cased' if 'cased' in m.lower() and 'uncased' not in m.lower() else 'Uncased')
        
        pivot = df.pivot_table(index=['Family', 'Question'], columns='Casing', values='HasAns Span F1').reset_index()
        
        if 'Cased' not in pivot.columns or 'Uncased' not in pivot.columns:
            create_placeholder_plot(filename, 'Case Sensitivity: Cased vs. Uncased Models', "Insufficient cased/uncased pairs.")
            return
            
        pivot = pivot.dropna(subset=['Cased', 'Uncased'])
        if pivot.empty:
            create_placeholder_plot(filename, 'Case Sensitivity: Cased vs. Uncased Models', "No complete cased/uncased pairs.")
            return

        family_order = ['BERT-base', 'ConfliBERT (Continual)', 'ConfliBERT (Scratch)']
        unique_qs = sorted(pivot['Question'].unique())

        if len(unique_qs) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 7.0), sharey=True)
            titles = {
                1: 'Question 1: Weapons & Force Extraction',
                2: 'Question 2: Infrastructure Damage Extraction'
            }

            for idx, q_val in enumerate(unique_qs[:2]):
                ax = axes[idx]
                q_sub = pivot[pivot['Question'] == q_val].copy()
                q_sub['sort_idx'] = q_sub['Family'].apply(lambda f: family_order.index(f) if f in family_order else 99)
                q_sub = q_sub.sort_values('sort_idx').reset_index(drop=True)

                x = np.arange(len(q_sub))
                w = 0.32
                r1 = ax.bar(x - w/2, q_sub['Uncased'], w, label='Uncased (Lowercase only)', color='#e74c3c')
                r2 = ax.bar(x + w/2, q_sub['Cased'], w, label='Cased (Preserves Capitalization)', color='#2980b9')

                ax.set_title(titles.get(q_val, f'Question {int(q_val)}'), fontsize=12.5, pad=12, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(q_sub['Family'], fontsize=11)
                ax.set_ylim(0, 1.0)
                ax.set_yticks(np.arange(0, 1.1, 0.2))
                ax.grid(True, axis='y', linestyle=':', alpha=0.6)
                ax.legend(loc='upper right', framealpha=0.9)

                if idx == 0:
                    ax.set_ylabel('Exact Span Match (Span F1)', fontsize=11)

                for bar in list(r1) + list(r2):
                    h = bar.get_height()
                    ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2, h),
                                ha='center', va='bottom', fontsize=9.5, xytext=(0, 3), textcoords='offset points')

            fig.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.95, wspace=0.18)
        else:
            fig, ax = plt.subplots(figsize=(12, 7.0))
            q_val = unique_qs[0]
            q_sub = pivot[pivot['Question'] == q_val].copy()
            q_sub['sort_idx'] = q_sub['Family'].apply(lambda f: family_order.index(f) if f in family_order else 99)
            q_sub = q_sub.sort_values('sort_idx').reset_index(drop=True)

            x = np.arange(len(q_sub))
            w = 0.32
            r1 = ax.bar(x - w/2, q_sub['Uncased'], w, label='Uncased (Lowercase only)', color='#e74c3c')
            r2 = ax.bar(x + w/2, q_sub['Cased'], w, label='Cased (Preserves Capitalization)', color='#2980b9')

            ax.set_title(f'Question {int(q_val)}: Cased vs. Uncased Tokenization', fontsize=12.5, pad=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(q_sub['Family'], fontsize=11)
            ax.set_ylabel('Exact Span Match (Span F1)', fontsize=11)
            ax.set_ylim(0, 1.0)
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            ax.grid(True, axis='y', linestyle=':', alpha=0.6)
            ax.legend(loc='upper right', framealpha=0.9)

            for bar in list(r1) + list(r2):
                h = bar.get_height()
                ax.annotate(f'{h:.3f}', (bar.get_x() + bar.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=9.5, xytext=(0, 3), textcoords='offset points')
            plt.tight_layout()

        fig.suptitle('Case Sensitivity: Cased vs. Uncased Models\n(Impact of preserving capitalization on acronyms & entity spans)', fontsize=14, y=0.97)
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_bert_casing_impact]: {e}\n{traceback.format_exc()}")

def plot_head_to_head_showdown(g_agg, filename="fig5_head_to_head_showdown.png"):
    """Compares top variants of ConfliBERT, BERT-base, and Generative LLMs ranked by HasAns bertscore_f1."""
    import traceback
    try:
        _ensure_assets_dir()
        if g_agg.empty: return

        best_rows = []
        for model in g_agg['Model_Raw'].unique():
            m_df = g_agg[g_agg['Model_Raw'] == model]
            best = m_df.sort_values(by='HasAns BERTScore', ascending=False).iloc[0]
            
            if 'confli' in model.lower():
                category = 'ConfliBERT (Extractive)'
                color = '#2b5c8f'
            elif 'bert' in model.lower():
                category = 'BERT-base (Extractive)'
                color = '#7f8c8d'
            else:
                category = 'Generative LLMs'
                color = '#9b59b6'
                
            best_rows.append({
                'Model': f"{best['Model_Raw']} [{best['Method']}]",
                'Score': best['HasAns BERTScore'],
                'Span F1': best['HasAns Span F1'],
                'Category': category,
                'Color': color
            })
            
        plot_df = pd.DataFrame(best_rows).sort_values(by='Score', ascending=True)
        if plot_df.empty: return

        fig, ax = plt.subplots(figsize=(12, 7.0))
        bars = ax.barh(plot_df['Model'], plot_df['Score'], color=plot_df['Color'], alpha=0.85, height=0.6)
        
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}",
                    va='center', fontsize=9, weight='bold')

        ax.set_xlim(0, 1.0)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_xlabel('Extraction Quality (BERTScore F1)', fontsize=11)
        ax.set_ylabel('Model & Setup', fontsize=11)
        ax.set_title('Overall Ranking: ConfliBERT vs. Generative LLMs\n(Best configuration per model ranked by BERTScore F1)', fontsize=14, pad=12)
        ax.grid(True, axis='x', linestyle=':', alpha=0.6)
        
        legend_handles = [
            mlines.Line2D([], [], color='#2b5c8f', marker='s', linestyle='None', markersize=10, label='ConfliBERT (Extractive)'),
            mlines.Line2D([], [], color='#7f8c8d', marker='s', linestyle='None', markersize=10, label='BERT-base (Extractive)'),
            mlines.Line2D([], [], color='#9b59b6', marker='s', linestyle='None', markersize=10, label='Generative LLMs')
        ]
        ax.legend(handles=legend_handles, loc='lower right')

        plt.tight_layout()
        plt.savefig(str(ASSETS_DIR / filename), dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"\n[ERROR in plot_head_to_head_showdown]: {e}\n{traceback.format_exc()}")
