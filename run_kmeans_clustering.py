"""
K-Means Clustering Analysis on Phase 1 Embeddings

This script:
1. Loads the Phase 1 pretrained model
2. Generates embeddings for resumes and jobs
3. Runs K-means clustering
4. Visualizes the clusters using t-SNE/PCA
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from collections import Counter

# Configuration
PHASE1_CHECKPOINT = "phase1_pretraining/best_checkpoint.pt"
DATA_FILE = "preprocess/data_without_augmentation_training.jsonl"
OUTPUT_DIR = "clustering_analysis"
N_CLUSTERS_RANGE = range(2, 15)  # Test different k values


def load_data(data_file: str):
    """Load the training data."""
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def extract_text(data):
    """Extract resume and job text from the data."""
    resume_texts = []
    job_texts = []
    labels = []
    job_titles = []
    resume_roles = []
    
    for item in data:
        # Extract resume text
        resume = item.get('resume', {})
        resume_text_parts = []
        resume_text_parts.append(f"Role: {resume.get('role', '')}")
        resume_text_parts.append(f"Experience Level: {resume.get('experience_level', '')}")
        
        # Skills
        skills = resume.get('skills', [])
        skill_names = [s.get('name', '') for s in skills if s.get('name')]
        if skill_names:
            resume_text_parts.append(f"Skills: {', '.join(skill_names)}")
        
        # Experience description
        for exp in resume.get('experience', []):
            for desc in exp.get('description', []):
                if isinstance(desc, dict) and desc.get('description'):
                    resume_text_parts.append(desc['description'][:500])  # Truncate long descriptions
        
        resume_text = ' '.join(resume_text_parts)
        resume_texts.append(resume_text)
        resume_roles.append(resume.get('role', 'unknown'))
        
        # Extract job text
        job = item.get('job', {})
        job_text_parts = []
        job_text_parts.append(f"Title: {job.get('title', '')}")
        job_text_parts.append(f"Experience Level: {job.get('experience_level', '')}")
        
        job_desc = job.get('description', {})
        if isinstance(job_desc, dict):
            job_text_parts.append(job_desc.get('original', ''))
        elif isinstance(job_desc, str):
            job_text_parts.append(job_desc)
        
        job_text = ' '.join(job_text_parts)
        job_texts.append(job_text)
        job_titles.append(job.get('title', 'unknown'))
        
        # Label
        label = item.get('metadata', {}).get('label', item.get('label', 0))
        if isinstance(label, str):
            label = 1 if label == 'positive' else 0
        labels.append(label)
    
    return resume_texts, job_texts, labels, job_titles, resume_roles


def generate_embeddings(texts, model):
    """Generate embeddings using the sentence transformer model."""
    print(f"Generating embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings


def find_optimal_k(embeddings, k_range, embedding_type=""):
    """Find optimal k using elbow method and silhouette score."""
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    print(f"\nFinding optimal k for {embedding_type} embeddings...")
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(embeddings)
        inertias.append(kmeans.inertia_)
        
        if k > 1:
            sil_score = silhouette_score(embeddings, kmeans.labels_)
            cal_score = calinski_harabasz_score(embeddings, kmeans.labels_)
            silhouette_scores.append(sil_score)
            calinski_scores.append(cal_score)
            print(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil_score:.4f}, Calinski-Harabasz={cal_score:.2f}")
        else:
            silhouette_scores.append(0)
            calinski_scores.append(0)
    
    return inertias, silhouette_scores, calinski_scores


def run_kmeans(embeddings, n_clusters):
    """Run K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    return kmeans, cluster_labels


def reduce_dimensions(embeddings, method='tsne'):
    """Reduce dimensions for visualization."""
    if method == 'tsne':
        print("Running t-SNE dimensionality reduction...")
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
    else:
        print("Running PCA dimensionality reduction...")
        reducer = PCA(n_components=2, random_state=42)
    
    reduced = reducer.fit_transform(embeddings)
    return reduced


def plot_clusters(reduced_embeddings, cluster_labels, true_labels, title, output_path, categories=None):
    """Plot clusters with both cluster assignments and true labels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: K-means clusters
    scatter1 = axes[0].scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=cluster_labels, 
        cmap='tab10', 
        alpha=0.6,
        s=20
    )
    axes[0].set_title(f'{title} - K-means Clusters')
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    
    # Plot 2: True labels (positive/negative)
    colors = ['red' if l == 0 else 'green' for l in true_labels]
    axes[1].scatter(
        reduced_embeddings[:, 0], 
        reduced_embeddings[:, 1], 
        c=colors, 
        alpha=0.6,
        s=20
    )
    axes[1].set_title(f'{title} - True Labels (Green=Positive, Red=Negative)')
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_path}")


def plot_elbow(k_range, inertias, silhouette_scores, title, output_path):
    """Plot elbow curve and silhouette scores."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    axes[0].plot(list(k_range), inertias, 'bo-')
    axes[0].set_xlabel('Number of Clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title(f'{title} - Elbow Method')
    axes[0].grid(True)
    
    # Silhouette plot
    axes[1].plot(list(k_range)[1:], silhouette_scores[1:], 'go-')
    axes[1].set_xlabel('Number of Clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title(f'{title} - Silhouette Score')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved elbow plot to {output_path}")


def analyze_clusters(cluster_labels, categories, category_name):
    """Analyze what categories are in each cluster."""
    print(f"\n{category_name} distribution per cluster:")
    
    cluster_to_categories = {}
    for cluster_id in sorted(set(cluster_labels)):
        indices = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
        cats_in_cluster = [categories[i] for i in indices]
        cat_counts = Counter(cats_in_cluster)
        cluster_to_categories[cluster_id] = cat_counts
        
        print(f"\n  Cluster {cluster_id} ({len(indices)} samples):")
        for cat, count in cat_counts.most_common(5):
            print(f"    {cat}: {count} ({100*count/len(indices):.1f}%)")
    
    return cluster_to_categories


def main():
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(DATA_FILE)
    print(f"Loaded {len(data)} samples")
    
    # Extract text
    resume_texts, job_texts, labels, job_titles, resume_roles = extract_text(data)
    print(f"Extracted {len(resume_texts)} resumes and {len(job_texts)} jobs")
    
    # Load sentence transformer model (base model used in Phase 1)
    print("\nLoading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings
    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS")
    print("="*60)
    
    resume_embeddings = generate_embeddings(resume_texts, model)
    job_embeddings = generate_embeddings(job_texts, model)
    
    # Combined embeddings (resume + job concatenated or averaged)
    combined_embeddings = (resume_embeddings + job_embeddings) / 2
    
    print(f"\nResume embeddings shape: {resume_embeddings.shape}")
    print(f"Job embeddings shape: {job_embeddings.shape}")
    print(f"Combined embeddings shape: {combined_embeddings.shape}")
    
    # Find optimal k for each embedding type
    print("\n" + "="*60)
    print("FINDING OPTIMAL NUMBER OF CLUSTERS")
    print("="*60)
    
    # Resume embeddings
    res_inertias, res_silhouettes, res_calinski = find_optimal_k(
        resume_embeddings, N_CLUSTERS_RANGE, "Resume"
    )
    plot_elbow(N_CLUSTERS_RANGE, res_inertias, res_silhouettes, 
               "Resume Embeddings", output_dir / "resume_elbow.png")
    
    # Job embeddings
    job_inertias, job_silhouettes, job_calinski = find_optimal_k(
        job_embeddings, N_CLUSTERS_RANGE, "Job"
    )
    plot_elbow(N_CLUSTERS_RANGE, job_inertias, job_silhouettes,
               "Job Embeddings", output_dir / "job_elbow.png")
    
    # Combined embeddings
    comb_inertias, comb_silhouettes, comb_calinski = find_optimal_k(
        combined_embeddings, N_CLUSTERS_RANGE, "Combined"
    )
    plot_elbow(N_CLUSTERS_RANGE, comb_inertias, comb_silhouettes,
               "Combined Embeddings", output_dir / "combined_elbow.png")
    
    # Find best k based on silhouette score
    best_k_resume = list(N_CLUSTERS_RANGE)[1:][np.argmax(res_silhouettes[1:])] 
    best_k_job = list(N_CLUSTERS_RANGE)[1:][np.argmax(job_silhouettes[1:])]
    best_k_combined = list(N_CLUSTERS_RANGE)[1:][np.argmax(comb_silhouettes[1:])]
    
    print(f"\nBest k (by silhouette score):")
    print(f"  Resume: k={best_k_resume} (silhouette={max(res_silhouettes[1:]):.4f})")
    print(f"  Job: k={best_k_job} (silhouette={max(job_silhouettes[1:]):.4f})")
    print(f"  Combined: k={best_k_combined} (silhouette={max(comb_silhouettes[1:]):.4f})")
    
    # Run K-means with optimal k
    print("\n" + "="*60)
    print("RUNNING K-MEANS CLUSTERING")
    print("="*60)
    
    # Also run with k matching number of job titles (33)
    k_job_titles = min(len(set(job_titles)), 33)
    
    for k in [best_k_resume, best_k_combined, k_job_titles]:
        print(f"\n--- K={k} ---")
        
        # Resume clustering
        _, resume_clusters = run_kmeans(resume_embeddings, k)
        
        # Job clustering  
        _, job_clusters = run_kmeans(job_embeddings, k)
        
        # Combined clustering
        _, combined_clusters = run_kmeans(combined_embeddings, k)
        
        # Analyze clusters
        analyze_clusters(resume_clusters, resume_roles, f"Resume roles (k={k})")
        analyze_clusters(job_clusters, job_titles, f"Job titles (k={k})")
        
        # Check label distribution in clusters
        print(f"\n  Positive/Negative distribution in combined clusters (k={k}):")
        for cluster_id in sorted(set(combined_clusters)):
            indices = [i for i, c in enumerate(combined_clusters) if c == cluster_id]
            pos_count = sum(1 for i in indices if labels[i] == 1)
            neg_count = len(indices) - pos_count
            print(f"    Cluster {cluster_id}: {len(indices)} samples - Pos: {pos_count} ({100*pos_count/len(indices):.1f}%), Neg: {neg_count} ({100*neg_count/len(indices):.1f}%)")
    
    # Dimensionality reduction and visualization
    print("\n" + "="*60)
    print("VISUALIZING CLUSTERS")
    print("="*60)
    
    # t-SNE for resume embeddings
    resume_tsne = reduce_dimensions(resume_embeddings, 'tsne')
    _, resume_clusters = run_kmeans(resume_embeddings, best_k_resume)
    plot_clusters(resume_tsne, resume_clusters, labels, 
                  f"Resume Embeddings (k={best_k_resume})",
                  output_dir / "resume_clusters_tsne.png",
                  resume_roles)
    
    # t-SNE for job embeddings
    job_tsne = reduce_dimensions(job_embeddings, 'tsne')
    _, job_clusters = run_kmeans(job_embeddings, best_k_job)
    plot_clusters(job_tsne, job_clusters, labels,
                  f"Job Embeddings (k={best_k_job})",
                  output_dir / "job_clusters_tsne.png",
                  job_titles)
    
    # t-SNE for combined embeddings
    combined_tsne = reduce_dimensions(combined_embeddings, 'tsne')
    _, combined_clusters = run_kmeans(combined_embeddings, best_k_combined)
    plot_clusters(combined_tsne, combined_clusters, labels,
                  f"Combined Embeddings (k={best_k_combined})",
                  output_dir / "combined_clusters_tsne.png")
    
    # Also visualize with k=33 (matching job titles)
    _, combined_clusters_33 = run_kmeans(combined_embeddings, k_job_titles)
    plot_clusters(combined_tsne, combined_clusters_33, labels,
                  f"Combined Embeddings (k={k_job_titles}, matching job titles)",
                  output_dir / "combined_clusters_k33_tsne.png")
    
    # Save results summary
    results = {
        "total_samples": len(data),
        "unique_resumes": len(set(resume_roles)),
        "unique_jobs": len(set(job_titles)),
        "embedding_dim": resume_embeddings.shape[1],
        "best_k_resume": best_k_resume,
        "best_k_job": best_k_job,
        "best_k_combined": best_k_combined,
        "silhouette_scores": {
            "resume": dict(zip([int(k) for k in N_CLUSTERS_RANGE], [float(s) for s in res_silhouettes])),
            "job": dict(zip([int(k) for k in N_CLUSTERS_RANGE], [float(s) for s in job_silhouettes])),
            "combined": dict(zip([int(k) for k in N_CLUSTERS_RANGE], [float(s) for s in comb_silhouettes]))
        }
    }
    
    with open(output_dir / "clustering_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n" + "="*60)
    print("CLUSTERING ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nResults saved to {output_dir}/")
    print(f"  - resume_elbow.png")
    print(f"  - job_elbow.png")
    print(f"  - combined_elbow.png")
    print(f"  - resume_clusters_tsne.png")
    print(f"  - job_clusters_tsne.png")
    print(f"  - combined_clusters_tsne.png")
    print(f"  - combined_clusters_k33_tsne.png")
    print(f"  - clustering_results.json")


if __name__ == "__main__":
    main()
