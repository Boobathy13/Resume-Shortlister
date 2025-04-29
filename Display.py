from prettytable import PrettyTable

def display_results(df, job_desc_weights=None):
    """Display results with JD skill highlighting"""
    if df.empty:
        print("No valid resumes processed")
        return

    # Sort by Combined_Score
    df = df.sort_values('Combined_Score', ascending=False)

    # Simplified ranking table (just rank, name, score)
    print("\n=== Candidate Ranking ===")
    simple_table = PrettyTable()
    simple_table.field_names = ["Rank", "Candidate", "Combined_Score"]
    simple_table.align = "l"

    for i, (_, row) in enumerate(df.iterrows(), 1):
        simple_table.add_row([
            i,
            row['Candidate'],
            f"{row['Combined_Score']:.1f}"
        ])
    print(simple_table)

    # Create filtered table with only candidates who have JD matches
    if job_desc_weights:
        matched_df = df[df['JD_Matched_Skills'].apply(lambda x: len(x) > 0)]

        if not matched_df.empty:
            print("\n=== Ranking only the Matched Candidates ===")
            matched_table = PrettyTable()
            matched_table.field_names = [
                "Rank", "Candidate", "Total", "JD Matches", "Matched Skills",
                "Exp", "Skills"
            ]

            matched_table.align = "l"

            for i, (_, row) in enumerate(matched_df.iterrows(), 1):
                matched_skills = ", ".join([
                    f"{skill}"
                    for skill in row['JD_Matched_Skills'].keys()
                ])

                matched_table.add_row([
                    i,
                    row['Candidate'],
                    f"{row['Combined_Score']:.1f}",
                    len(row['JD_Matched_Skills']),
                    matched_skills,
                    row['Experience (years)'],
                    sum(row['Skills'].values())
                ])

            print(matched_table)

            # Simplified matched candidates ranking
            print("\n=== Shortlisted Candidates ===")
            simple_matched_table = PrettyTable()
            simple_matched_table.field_names = ["Rank", "Candidate", "Combined_Score"]
            simple_matched_table.align = "l"

            for i, (_, row) in enumerate(matched_df.iterrows(), 1):
                simple_matched_table.add_row([
                    i,
                    row['Candidate'],
                    f"{row['Combined_Score']:.1f}"
                ])
            print(simple_matched_table)
        else:
            print("\nNo candidates matched the specified skills")

    # Score breakdown for top candidates
    print("\n=== Score Breakdown for All Candidates ===")
    score_table = PrettyTable()
    score_table.field_names = [
                                  "Rank", "Candidate", "Total", "Exp", "Skills",
                                  "Certs", "Projects", "Roles"
                              ] + (["JD Matches"] if job_desc_weights else [])

    score_table.align = "l"

    for i, (_, row) in enumerate(df.head(5).iterrows(), 1):
        row_data = [
        i,
        row['Candidate'],
        f"{row['Combined_Score']:.1f}",
        f"{row['Experience']:.1f}",
        f"{sum(row['Skills'].values())}",
        f"{row['Certifications']}",
        f"{row['Projects']}",
        f"{row['Role_Relevance']:.1f}"
        ]


        if job_desc_weights:
            row_data.append(len(row['JD_Matched_Skills']))

        score_table.add_row(row_data)

    print(score_table)