from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import os
import pandas as pd
from prettytable import PrettyTable
import spacy
import re
from Extract_data import get_job_description_skills
from Extract_data import extract_text_from_pdf, detect_experience, count_certifications, count_projects, extract_skills
from Extract_data import RoleAnalyzer
from Extract_data import calculate_scores
from Extract_data import SCORE_WEIGHTS, KEYWORD_FILE       


class HybridResumeAnalyzer:
    def __init__(self):
        # Load SBERT model (using a lightweight general-purpose model)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize your existing components
        self.nlp = spacy.load("en_core_web_lg")
        self.role_analyzer = RoleAnalyzer()

        try:
            self.keyword_df = pd.read_csv(KEYWORD_FILE)
        except Exception as e:
            print(f"Error loading keyword file: {e}")
            self.keyword_df = pd.DataFrame()

    def enhanced_process_resumes(self, resume_dir, job_description=None, job_desc_weights=None):
        """Process resumes with both rule-based and SBERT analysis"""
        resume_files = [os.path.join(resume_dir, f)
                        for f in os.listdir(resume_dir)
                        if os.path.isfile(os.path.join(resume_dir, f)) and f.lower().endswith('.pdf')]

        # Encode job description if provided
        jd_embedding = None
        if job_description:
            jd_embedding = self.sbert_model.encode(job_description, convert_to_tensor=True)

        results = []
        for file_path in resume_files:
            try:
                # Extract text and basic features (existing functionality)
                text = extract_text_from_pdf(file_path)
                if not text.strip():
                    continue

                candidate_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[0].title()

                # Get rule-based features
                rule_based_features = {
                    'experience': detect_experience(text),
                    'certifications': count_certifications(text),
                    'projects': count_projects(text),
                    'skills': extract_skills(text, self.keyword_df),
                    'role_relevance': self.role_analyzer.calculate_role_score(
                        self.role_analyzer.extract_roles(text)
                    )['normalized_score']
                }

                # Calculate SBERT similarity if JD provided
                sbert_score = 0
                similarity_details = {}
                if jd_embedding is not None:
                    # Split resume into meaningful chunks (e.g., sections or sentences)
                    resume_chunks = self._split_resume_into_chunks(text)
                    chunk_embeddings = self.sbert_model.encode(resume_chunks, convert_to_tensor=True)

                    # Calculate similarities
                    similarities = util.pytorch_cos_sim(jd_embedding, chunk_embeddings)[0]
                    sbert_score = torch.max(similarities).item() * 100  # Convert to percentage

                    # Get top 3 most relevant chunks
                    top_indices = torch.topk(similarities, min(3, len(similarities))).indices
                    similarity_details = {
                        'top_matches': [(resume_chunks[i], similarities[i].item())
                                        for i in top_indices],
                        'max_similarity': sbert_score,
                        'mean_similarity': torch.mean(similarities).item() * 100
                    }

                # Calculate combined score (adjust weights as needed)
                rule_based_scores = calculate_scores(rule_based_features, SCORE_WEIGHTS, job_desc_weights)
                combined_score = rule_based_scores['total'] * 0.7 + sbert_score * 0.3

                # Relevance-based scoring for job description skills
                relevance_score = 0
                if job_description:
                    jd_keywords = [kw.strip().lower() for kw in job_description.split(',')]
                    for skill, count in rule_based_features['skills'].items():
                        if any(jd_kw in skill.lower() for jd_kw in jd_keywords):
                            relevance_score += count * 3  # Triple weight for relevant skills
                        else:
                            relevance_score += count * 0.5  # Lower weight for unrelated skills

                combined_score += relevance_score  # Add relevance-based score to the combined score

                # Boost scores for popular tech skills matching the job description
                tech_boost = 0
                popular_tech_skills = [
                    'java', 'python', 'javascript', 'react', 'angular', 'nodejs',
                    'express', 'mongodb', 'mysql', 'aws', 'azure', 'docker', 'kubernetes',
                    'spring', 'flask', 'django', 'typescript', 'vue', 'c++', 'html', 'css'
                ]
                for skill in rule_based_features['skills'].keys():
                    if skill.lower() in popular_tech_skills:
                        tech_boost += rule_based_features['skills'][skill] * 1.5  # 1.5x weight for popular tech skills

                combined_score += tech_boost  # Add the boost for popular tech skills

                # Tie-breaking logic for candidates with the same skill level
                if job_description:
                    jd_keywords = [kw.strip().lower() for kw in job_description.split(',')]
                    for skill, count in rule_based_features['skills'].items():
                        if any(jd_kw in skill.lower() for jd_kw in jd_keywords):
                            combined_score += count * 0.1  # Add a small boost for matching skills to break ties

                # Ensure candidates with the same skill level are ranked by other factors
                combined_score += (
                    rule_based_features['experience'] * 0.05 +
                    rule_based_features['certifications'] * 0.03 +
                    rule_based_features['projects'] * 0.02
                )

                results.append({
                    'Candidate': candidate_name,
                    'File': file_path,
                    'Rule_Based_Score': rule_based_scores['total'],
                    'SBERT_Score': sbert_score,
                    'Combined_Score': combined_score,
                    'Experience': rule_based_features['experience'],
                    'Certifications': rule_based_features['certifications'],
                    'Projects': rule_based_features['projects'],
                    'Skills': rule_based_features['skills'],
                    'Role_Relevance': rule_based_features['role_relevance'],
                    'Similarity_Details': similarity_details,
                    'JD_Matched_Skills': {}, 
                    **rule_based_scores
                })

            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")

        return pd.DataFrame(results)

    def _split_resume_into_chunks(self, text, max_chunk_length=300):
        """Split resume text into meaningful chunks for SBERT processing"""
        # First try to split by sections
        sections = re.split(r'\n\s*\n|\b(?:experience|education|projects|skills):?\b', text, flags=re.IGNORECASE)
        chunks = []

        for section in sections:
            if not section.strip():
                continue

            # If section is too long, split into sentences
            if len(section) > max_chunk_length:
                sentences = [sent.text for sent in self.nlp(section).sents]
                current_chunk = ""
                for sent in sentences:
                    if len(current_chunk) + len(sent) < max_chunk_length:
                        current_chunk += " " + sent
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = sent
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
            else:
                chunks.append(section.strip())

        return chunks


def enhanced_display_results(df, job_description=None):
    """Enhanced display showing both rule-based and SBERT results"""
    if df.empty:
        print("No valid resumes processed")
        return

    # Sort by combined score
    df = df.sort_values('Combined_Score', ascending=False)

    # Main results table
    print("\n=== Candidate Ranking ===")
    main_table = PrettyTable()
    main_table.field_names = [
        "Rank", "Candidate", "Combined", "Rule-Based", "SBERT",
        "Exp", "Certs", "Projects", "Skills"
    ]
    main_table.align = "l"

    for i, (_, row) in enumerate(df.iterrows(), 1):
        main_table.add_row([
            i,
            row['Candidate'],
            f"{row['Combined_Score']:.1f}",
            f"{row['Rule_Based_Score']:.1f}",
            f"{row['SBERT_Score']:.1f}" if row['SBERT_Score'] else "N/A",
            f"{row['Experience']:.1f}y",
            row['Certifications'],
            row['Projects'],
            sum(row['Skills'].values())
        ])
    print(main_table)

    import textwrap

    # Show semantic matches if JD was provided
    if job_description and 'Similarity_Details' in df.columns:
        print("\n=== Semantic Matches with Job Description ===")
        for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
            if not row['Similarity_Details']:
                continue

            print(f"\n#{i} {row['Candidate']} (Similarity: {row['SBERT_Score']:.1f}%)")
            for j, (chunk, score) in enumerate(row['Similarity_Details']['top_matches'], 1):
                print(f"\nMatch {j} ({score * 100:.1f}%):")
                print(textwrap.fill(chunk, width=80))