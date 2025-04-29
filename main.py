from Score import HybridResumeAnalyzer
from Display import display_results
from Extract_data import get_job_description_skills
import pandas as pd
import matplotlib.pyplot as plt
from Extract_data import RESUME_DIR


def main():
    print("Advanced Resume Analyzer with SBERT Integration")

    # Initialize analyzer
    analyzer = HybridResumeAnalyzer()

    # Get job description (optional)
    job_description = None
    print("\nEnter job description (or press Enter to skip):")
    jd_text = input().strip()
    if jd_text:
        job_description = jd_text

    # Get prioritized skills (optional)
    job_desc_weights = None
    if job_description:
        print("\nWould you like to specify additional prioritized skills? (y/n)")
        if input().strip().lower() == 'y':
            job_desc_weights = get_job_description_skills()

    # Process resumes with job description influencing scores
    results_df = analyzer.enhanced_process_resumes(
        RESUME_DIR,
        job_description=job_description,  # Pass job description
        job_desc_weights=job_desc_weights  # Pass prioritized skills
    )

    # Normalize combined scores to ensure they are less than 100 and realistic
    results_df['Combined_Score'] = results_df['Combined_Score'].apply(lambda x: min(x, 95))

    # Display results
    if not results_df.empty:
        display_results(results_df, job_desc_weights)

        # Visualization - Top Candidates Skill Distribution
        top_candidates = results_df.sort_values('Combined_Score', ascending=False).head(5)

        if not top_candidates.empty:
            skills_data = pd.DataFrame(top_candidates['Skills'].tolist(), index=top_candidates['Candidate'])

            # Ensure skills_data is cleaned and filled
            skills_data = skills_data.dropna(how='all')  # Drop rows with all NaNs
            skills_data = skills_data.fillna(0)          # Fill missing values with 0
            skills_data = skills_data.astype(int)        # Convert all values to integers

            # Check if skills_data is empty before generating any plots
            if skills_data.empty:
                print("No skill data available for visualization.")
                return

            # Ensure a minimum of 10 diverse verticals for visualization
            skill_totals = skills_data.sum(axis=0).sort_values(ascending=False)
            if len(skill_totals) > 10:
                # Limit to the top 10 skill categories
                top_skills = skill_totals.head(10).index
                skills_data = skills_data[top_skills]
            else:
                # Use all available skill categories if fewer than 10
                top_skills = skill_totals.index

            # Replace 'Development' with 'Dev' in skill labels for easier usage
            skills_data.columns = [col.replace('Development', 'Dev') for col in skills_data.columns]

            # Generate a single image with both bar chart and pie chart
            fig, axes = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1]})

            # Display bar chart on the first subplot
            skills_data.plot(kind='barh', stacked=True, colormap='viridis', ax=axes[0])
            axes[0].set_title('Skill Distribution - Top Candidates', fontsize=16)
            axes[0].set_xlabel('Skill Count', fontsize=12)
            axes[0].set_ylabel('Candidate', fontsize=12)
            axes[0].legend(title='Skills', bbox_to_anchor=(1.05, 1), loc='upper left')

            # Display pie chart on the second subplot
            skill_totals = skills_data.sum(axis=0)
            skill_totals = skill_totals[skill_totals > 0]  # Filter out skills with zero count
            if not skill_totals.empty:
                wedges, texts, autotexts = axes[1].pie(
                    skill_totals, labels=skill_totals.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab20.colors, textprops={'fontsize': 10}
                )
                axes[1].set_title('Overall Skill Category Distribution', fontsize=16)

                # Adjust text visibility for pie chart
                for text in texts:
                    text.set_fontsize(9)
                for autotext in autotexts:
                    autotext.set_fontsize(8)

            # Adjust layout to ensure equal space for both plots
            plt.subplots_adjust(wspace=0.3)  # Adjust space between the two plots

            # Save the combined image
            plt.tight_layout(rect=[0, 0, 1, 1])
            plt.savefig('combined_visualization.png')
            plt.show()
        else:
            print("No top candidates to plot skills for.")
    else:
        print("No valid resumes processed.")

if __name__ == '__main__':
    main()

    # Optional cleanup (especially useful if using GPU)
    import torch
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    # Force exit cleanly
    import sys
    sys.exit(0)
