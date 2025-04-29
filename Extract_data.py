import spacy
import PyPDF2
import os
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from dateutil.relativedelta import relativedelta

# Load English language model
nlp = spacy.load("en_core_web_lg")

# Configuration
RESUME_DIR = 'C:\BlitzkrieG\Projects\Final Year Project\Resumes'
KEYWORD_FILE = os.path.join(os.path.dirname(__file__), 'NLP4.csv')

# Scoring weights
SCORE_WEIGHTS = {
    "experience": 5,  # per year
    "certification": 3,  # per cert
    "project": 2,  # per project
    "role_relevance": 3,  # multiplier for role score
    "skills": {  # skill category weights
        "Statistics": 8,
        "Machine Learning": 9,
        "Deep Learning": 10,
        "R Language": 5,
        "Python Language": 5,
        "NLP": 10,
        "Data Engineering": 4,
        "Web Development": 2
    }
}


class RoleAnalyzer:
    def __init__(self):
        # Predefined weights for companies and roles
        self.company_weights = {
            "google": 1.5, "microsoft": 1.4, "amazon": 1.4, "facebook": 1.5,
            "apple": 1.4, "ibm": 1.3, "oracle": 1.2, "netflix": 1.4,
            "twitter": 1.3, "linkedin": 1.3, "nvidia": 1.4, "tesla": 1.4,
            "stanford": 1.3, "mit": 1.3, "berkeley": 1.2
        }

        self.role_weights = {
            "engineer": 1.3, "scientist": 1.5, "researcher": 1.4,
            "developer": 1.2, "manager": 1.3, "director": 1.6,
            "architect": 1.4, "specialist": 1.2, "intern": 0.7,
            "assistant": 0.8, "analyst": 1.1, "consultant": 1.2
        }

        self.role_keywords = {
            "machine learning": ["machine learning", "ml", "ai engineer", "deep learning"],
            "data science": ["data scientist", "data analyst", "data engineer"],
            "software": ["software engineer", "backend", "frontend", "full stack"],
            "research": ["research scientist", "research engineer", "research assistant"]
        }

        self.title_patterns = [
            r"(?P<title>[A-Za-z ]+?)\s(?:at|in|@)\s(?P<company>[A-Za-z0-9 &]+)",
            r"(?P<title>[A-Za-z ]+?),\s*(?:at\s*)?(?P<company>[A-Za-z0-9 &]+)",
            r"(?P<company>[A-Za-z0-9 &]+)\s*[-–]\s*(?P<title>[A-Za-z ]+)"
        ]

    def extract_roles(self, text):
        """Extract roles and companies from resume text"""
        doc = nlp(text)
        roles = []

        # Pattern matching for common title formats
        for pattern in self.title_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                roles.append({
                    'title': match.group('title').strip().lower(),
                    'company': match.group('company').strip().lower()
                })

        # NER-based extraction
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Look for surrounding text that might be a title
                start = max(0, ent.start_char - 50)
                end = min(len(text), ent.end_char + 50)
                context = text[start:end].lower()

                for pattern in self.title_patterns:
                    match = re.search(pattern, context, re.IGNORECASE)
                    if match:
                        roles.append({
                            'title': match.group('title').strip(),
                            'company': ent.text.lower()
                        })

        # Deduplicate roles
        unique_roles = []
        seen = set()
        for role in roles:
            key = (role['title'], role['company'])
            if key not in seen:
                seen.add(key)
                unique_roles.append(role)

        return unique_roles

    def calculate_role_score(self, roles):
        """Calculate weighted score based on roles and companies"""
        total_score = 0
        role_details = []

        for role in roles:
            # Calculate company weight
            company = role['company']
            company_weight = 1.0  # Default weight

            for known_company, weight in self.company_weights.items():
                if known_company in company:
                    company_weight = weight
                    break

            # Calculate role weight
            title = role['title']
            role_weight = 1.0  # Default weight

            # Check for specific role keywords
            for role_type, keywords in self.role_keywords.items():
                if any(keyword in title for keyword in keywords):
                    role_weight = 1.5  # Boost for relevant roles
                    break

            # Check for seniority indicators
            if "senior" in title:
                role_weight *= 1.3
            elif "junior" in title:
                role_weight *= 0.8
            elif any(word in title for word in ["lead", "principal", "head"]):
                role_weight *= 1.5
            elif any(word in title for word in ["associate", "assistant"]):
                role_weight *= 0.9

            # Apply base role weights
            for role_key, weight in self.role_weights.items():
                if role_key in title:
                    role_weight *= weight
                    break

            role_score = company_weight * role_weight
            total_score += role_score

            role_details.append({
                'title': role['title'],
                'company': role['company'],
                'score': role_score
            })

        return {
            'total_score': total_score,
            'role_details': role_details,
            'normalized_score': min(10, total_score)  # Cap at 10
        }

import layoutparser as lp

class ResumeProcessor:
    def __init__(self):
        self.model = lp.Detectron2LayoutModel(
            config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8]
        )


# Use self.model instead of loading each time
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ' '.join([page.extract_text() or '' for page in reader.pages])
            return text.replace('\n', ' ').replace('\r', ' ')  # Clean line breaks
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def detect_experience(text):
    """More robust experience detection with better date handling"""
    text = text.lower()
    total_years = 0

    # Method 1: Direct year mentions
    year_matches = re.findall(r'(\d+)\s+(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', text)
    if year_matches:
        total_years = sum(int(y) for y in year_matches)

    # Method 2: Employment duration - more robust pattern
    duration_pattern = r'''
        (?:                         
            \b                      
            (?:jan|feb|mar|apr|may|jun|  
             jul|aug|sep|oct|nov|dec|
             january|february|march|april|may|june|
             july|august|september|october|november|december)
            [a-z]*                  
            \s+                     
            (?:19|20)?\d{2}         
            \b                      
        )
    '''

    # Find all date-like patterns first
    date_pattern = re.compile(duration_pattern, re.VERBOSE | re.IGNORECASE)
    dates = date_pattern.findall(text)

    # Then look for date ranges
    range_pattern = re.compile(
        rf'({duration_pattern})\s*(?:-|to|–)\s*({duration_pattern}|present|current|now)',
        re.VERBOSE | re.IGNORECASE
    )

    # Handle overlapping date ranges by keeping track of unique periods
    unique_periods = []

    for start, end in range_pattern.findall(text):
        try:
            start_date = parse(start, fuzzy=True)
            if any(x in end.lower() for x in ['present', 'current', 'now']):
                end_date = datetime.now()
            else:
                end_date = parse(end, fuzzy=True)

            # Check for overlaps with existing periods
            overlap = False
            for period in unique_periods:
                if not (end_date < period[0] or start_date > period[1]):
                    overlap = True
                    break

            if not overlap:
                unique_periods.append((start_date, end_date))
                delta = relativedelta(end_date, start_date)
                total_years += delta.years + delta.months / 12
        except Exception as e:
            continue

    # Cap between 0-10 years
    return min(10, max(0, total_years))


def count_certifications(text):
    """Comprehensive certification detection with multiple validation methods"""
    text = text.replace('\n', ' ').lower()
    counts = []  # We'll track counts from all methods

    # Method 1: Original section-based counting (keep your existing approach)
    cert_section = re.search(
        r'(certifications?|licenses?|qualifications?|credentials|courses|certificates)(.*?)(?=(education|experience|projects|skills|\n\s*\n|$))',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if cert_section:
        section_text = cert_section.group(2)
        items = re.findall(r'(•|\d+\.|-\s|\[.\]\s|✓\s)(.*?)(?=(•|\d+\.|-\s|\[.\]\s|✓\s|$))', section_text)
        counts.append(len(items))

    # Method 2: Original pattern matching (keep your existing patterns)
    patterns = [
        r'\b(?:certified|licensed)\s+(?:in|as|for)\s+[a-z\s]+',
        r'\b(?:[a-z]+\s)?(?:certification|certificate|license|qualification)\b',
        r'\b(?:aws|google|microsoft|oracle|cisco)\s+certified\b',
        r'\b(?:passed|completed)\s+(?:the\s)?[a-z]+\s+(?:certification|exam)',
        r'\b(?:pmp|ccna|ccnp|awscsa|ocp|mcse|comptia)\b'
    ]
    verification_keywords = [
        'certification', 'certificate', 'license',
        'credential', 'accredited', 'validated'
    ]
    all_matches = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            context = text[max(0, match.start() - 20):min(len(text), match.end() + 20)]
            if any(kw in context for kw in verification_keywords):
                all_matches.add(match.group().strip())
    counts.append(len(all_matches))

    # Method 3: New - Certification abbreviations
    cert_abbreviations = {
        'pmp', 'ccna', 'ccnp', 'awscsa', 'ocp', 'mcse', 'comptia',
        'cissp', 'ceh', 'gcp', 'azure', 'ocpjp', 'ocpjd', 'rhce',
        'cfa', 'cfp', 'cia', 'cpa', 'cisco', 'aws', 'gcp', 'azure'
    }
    abbrev_matches = set()
    for abbrev in cert_abbreviations:
        if re.search(r'\b' + re.escape(abbrev) + r'\b', text):
            abbrev_matches.add(abbrev)
    counts.append(len(abbrev_matches))

    # Method 4: New - Certification issuing bodies
    issuing_bodies = {
        'aws', 'microsoft', 'oracle', 'cisco', 'google', 'comptia',
        'pmi', 'isc2', 'ecouncil', 'red hat', 'linux foundation',
        'sas', 'cloudera', 'databricks', 'snowflake', 'nptel'
    }
    body_matches = set()
    for body in issuing_bodies:
        if re.search(r'\b' + re.escape(body) + r'\b', text):
            body_matches.add(body)
    counts.append(len(body_matches))

    # Method 5: New - Certification-like dates (e.g., "Certified 2020")
    date_matches = len(re.findall(
        r'(?:certified|licensed|passed)\s+(?:in\s)?(?:19|20)\d{2}',
        text
    ))
    counts.append(date_matches)

    # Return the maximum count from all methods
    return min(20, max(counts))  # Cap at 20 certifications


def count_projects(text):
    """Robust project detection handling multiple resume formats"""
    text = re.sub(r'\s+', ' ', text.lower())
    counts = []

    # Method 1: Enhanced section detection with more header patterns
    project_headers = [
        r'(projects?|portfolio|initiatives?|work\s+experience|selected\s+work|personal\s+work)',
        r'(?:^|\n)\s*#+\s*projects?\s*#*',  # For markdown-style headers
        r'(?:^|\n)\s*projects?\s*[:]'  # For "Projects:" style headers
    ]

    for header_pattern in project_headers:
        project_section = re.search(
            f'{header_pattern}(.*?)(?=(education|experience|certifications|skills|\\n\\s*\\n|$))',
            text,
            re.IGNORECASE | re.DOTALL
        )
        if project_section:
            section_text = project_section.group(2)
            # Enhanced bullet point detection
            items = re.findall(
                r'(?:^|\n)\s*(?:[\•\-\*\+‣⁃]|\d+[\.\)]|\[[x\s]\]|✓)\s*(.+?)(?=\n\s*(?:[\•\-\*\+‣⁃]|\d+[\.\)]|\[[x\s]\]|✓|\w+\s+\d{4}|$))',
                section_text
            )
            counts.append(len(items))

    # Method 2: Technology-validated projects (expanded keywords)
    tech_keywords = [
        # Programming languages
        r'python', r'java', r'c\+\+', r'javascript', r'typescript',
        # Web technologies
        r'react', r'angular', r'vue', r'node\.?js', r'express',
        r'django', r'flask', r'spring', r'html', r'css',
        # Data/AI
        r'machine\s+learning', r'tensorflow', r'pytorch', r'keras',
        r'nlp', r'computer\s+vision', r'deep\s+learning',
        # Databases
        r'sql', r'mysql', r'postgres', r'mongodb', r'nosql',
        # Cloud/DevOps
        r'aws', r'azure', r'gcp', r'docker', r'kubernetes',
        # Other
        r'mern', r'full\s+stack', r'web\s+app', r'web\s+application'
    ]

    project_indicators = [
        r'\b(?:developed|created|built|implemented|designed|constructed|programmed|made)\b',
        r'\b(?:project|initiative|application|system|website|portal|platform)\b',
        r'\bgithub\.com/[a-z0-9-]+/[a-z0-9-]+\b'
    ]

    # Count project-like sentences with tech context
    project_sentences = 0
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        has_tech = any(re.search(tech, sentence) for tech in tech_keywords)
        has_indicator = any(re.search(ind, sentence) for ind in project_indicators)
        if has_tech and has_indicator:
            project_sentences += 1
    counts.append(project_sentences)

    # Method 3: Project title patterns (e.g., "Resume Shortlister - Web Application")
    title_matches = len(re.findall(
        r'(?:^|\n)\s*.+?\s*[-–]\s*(?:web\s+app|application|system|project)',
        text
    ))
    counts.append(title_matches)

    # Method 4: Date-bound projects (e.g., "Project Name (Jan 2020 - Present)")
    date_bound = len(re.findall(
        r'(?:^|\n)\s*.+?\s*\((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}\s*(?:to|-|–)\s*(?:present|now|current|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4})\)',
        text, re.IGNORECASE
    ))
    counts.append(date_bound)

    # Return the maximum count found by any method
    return min(20, max(counts)) if counts else 0

from collections import Counter
import re
from spacy.matcher import PhraseMatcher


def extract_skills(text, keyword_df):
    """Enhanced skill extraction with comma-separated list support"""
    matcher = PhraseMatcher(nlp.vocab)
    skill_counts = Counter()

    # Pre-process text to identify and emphasize skills sections
    text = preprocess_text(text)

    # First pass: Try to find explicit skills sections
    skills_sections = extract_sections_with_keywords(text, ['skills', 'technical skills', 'competencies'])

    # Combine main text with emphasized skills sections (appearing twice)
    processed_text = text + " " + " ".join(skills_sections) * 2

    # Build matcher patterns for all skill categories
    for column in keyword_df.columns:
        patterns = [
            nlp(skill.strip().lower())
            for skill in keyword_df[column].dropna()
            if skill.strip()
        ]
        if patterns:
            matcher.add(column, patterns)

    # Process text and count matches with enhanced matching
    doc = nlp(processed_text)

    # First handle comma-separated lists
    for sent in doc.sents:
        # Look for patterns like "Skills: Python, Java, TensorFlow"
        if is_skills_sentence(sent):
            for token in sent:
                if token.text == ',':
                    continue
                # Check if token matches any skill
                for column in keyword_df.columns:
                    if token.text.lower() in [s.lower() for s in keyword_df[column].dropna()]:
                        skill_counts[column] += 1

    # Then do the original phrase matching
    matches = matcher(doc)
    seen_spans = set()

    for match_id, start, end in matches:
        span = doc[start:end]

        # Skip if we've already counted this exact span
        if span.text.lower() in seen_spans:
            continue

        seen_spans.add(span.text.lower())

        # Additional context checks
        if is_valid_skill_context(span):
            skill_category = nlp.vocab.strings[match_id]
            skill_counts[skill_category] += 1

    return skill_counts


def is_skills_sentence(sent):
    """Check if sentence appears to be a skills list"""
    # Check for keywords indicating a skills section
    skill_keywords = ['skills', 'technical', 'expertise', 'proficient', 'competencies']
    if any(keyword in sent.text.lower() for keyword in skill_keywords):
        return True

    # Check for list-like structure with multiple commas
    if sum(1 for token in sent if token.text == ',') >= 2:
        return True

    return False


def preprocess_text(text):
    """Enhanced text preprocessing with comma handling"""
    # Normalize different comma representations
    text = re.sub(r'[,\;]', ' , ', text)
    # Normalize bullets and whitespace
    text = re.sub(r'[\•\-\*\+‣⁃]', ' • ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower()


def extract_sections_with_keywords(text, section_keywords):
    """Extract sections that likely contain skills lists"""
    doc = nlp(text.lower())
    sections = []

    # Look for section headers
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if any(keyword in sent_text for keyword in section_keywords):
            # Get the next 3 sentences (likely the skills list)
            next_sents = list(doc[sent.end:].sents)[:3]
            sections.append(" ".join([s.text for s in next_sents]))

    return sections


def is_valid_skill_context(span):
    """Verify the surrounding context looks like a skill listing"""
    # Check for bullet point patterns
    if re.search(r'^[\•\-\*\+]\s*', span.sent.text):
        return True

    # Check if in a list pattern (like comma separated)
    if len(span.sent) > 1 and any(t.text == ',' for t in span.sent):
        return True

    # Check if in a skills section
    if any(keyword in span.sent.text.lower()
           for keyword in ['skills', 'technical', 'competencies']):
        return True

    return False


def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Normalize bullets and whitespace
    text = re.sub(r'[\•\-\*\+‣⁃]', ' • ', text)
    text = re.sub(r'[,\;]', ' ', text)  # Treat commas/semicolons as separators
    text = re.sub(r'\s+', ' ', text)
    return text.lower()

def get_job_description_skills():
    """Get user input for prioritized skills (automatically assigns weight 10)"""
    print("\n=== Job Description Skills ===")
    print("Enter the skills you want to prioritize (comma separated):")
    user_input = input("Skills: ").strip()

    if not user_input:
        return None

    skills = [skill.strip().lower() for skill in user_input.split(',') if skill.strip()]
    return {skill: 10 for skill in skills}  # Automatically assign weight 10


def calculate_scores(data, base_weights, job_desc_weights=None):
    """Calculate scores with optional job description weighting"""
    scores = {
        'experience': data['experience'] * base_weights['experience'],
        'certifications': data['certifications'] * base_weights['certification'],
        'projects': data['projects'] * base_weights['project'],
        'role_relevance': data['role_relevance'] * base_weights['role_relevance']
    }

    # Calculate skill score with JD weighting
    skill_score = 0
    for skill, count in data['skills'].items():
        base_weight = base_weights['skills'].get(skill, 1)

        # Apply job description boost if skill matches
        jd_boost = 1
        if job_desc_weights:
            for jd_skill, jd_weight in job_desc_weights.items():
                if jd_skill.lower() in skill.lower():
                    jd_boost = jd_weight / 5  # Normalize to reasonable multiplier
                    break

        skill_score += min(3, count) * base_weight * jd_boost

    scores['skills'] = skill_score
    scores['total'] = sum(scores.values())

    return scores


def get_matched_jd_skills(candidate_skills, job_desc_weights):
    """Get skills that match job description with their weights"""
    if not job_desc_weights:
        return {}

    matched = {}
    for skill in candidate_skills:
        for jd_skill, weight in job_desc_weights.items():
            if jd_skill.lower() in skill.lower():
                matched[skill] = weight
                break
    return matched


def process_resumes(resume_dir, job_desc_weights=None):
    """Process resumes with optional job description weighting"""
    resume_files = [os.path.join(resume_dir, f)
                    for f in os.listdir(resume_dir)
                    if os.path.isfile(os.path.join(resume_dir, f)) and f.lower().endswith('.pdf')]

    try:
        keyword_df = pd.read_csv(KEYWORD_FILE)
    except Exception as e:
        print(f"Error loading keyword file: {e}")
        return pd.DataFrame()

    role_analyzer = RoleAnalyzer()
    results = []

    for file_path in resume_files:
        try:
            text = extract_text_from_pdf(file_path)
            if not text.strip():
                print(f"Skipping empty file: {file_path}")
                continue

            # Debugging: Log the extracted text for each resume
            print(f"Extracted text from {file_path}:")
            print(text[:500])  # Print the first 500 characters of the extracted text

            # Debugging: Log the extracted skills for each resume
            extracted_skills = extract_skills(text, keyword_df)
            print(f"Extracted skills from {file_path}:")
            print(extracted_skills)

            # Extract candidate name from filename
            candidate_name = os.path.splitext(os.path.basename(file_path))[0].split('_')[0].title()

            # Extract features
            experience = detect_experience(text)
            certifications = count_certifications(text)
            projects = count_projects(text)
            skills = extract_skills(text, keyword_df)
            role_analysis = role_analyzer.calculate_role_score(role_analyzer.extract_roles(text))

            # Get matched JD skills
            matched_jd_skills = get_matched_jd_skills(skills.keys(), job_desc_weights)

            # Calculate scores
            scores = calculate_scores(
                {
                    'experience': experience,
                    'certifications': certifications,
                    'projects': projects,
                    'skills': skills,
                    'role_relevance': role_analysis['normalized_score']
                },
                SCORE_WEIGHTS,
                job_desc_weights
            )

            results.append({
                'Candidate': candidate_name,
                'Experience (years)': round(experience, 1),
                'Certifications': certifications,
                'Projects': projects,
                'Skills': dict(skills),
                'JD_Matched_Skills': matched_jd_skills,
                'Roles': role_analysis['role_details'],
                'Role Score': scores['role_relevance'],
                'Experience Score': scores['experience'],
                'Certification Score': scores['certifications'],
                'Project Score': scores['projects'],
                'Skill Score': scores['skills'],
                'Total Score': scores['total'],
                'File': file_path
            })

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return pd.DataFrame(results)