import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate synthetic job dataset
def generate_job_dataset(n_samples=1000):
    # Define job roles and their corresponding skills
    job_roles = {
        'Data Scientist': [
            'python', 'r', 'sql', 'machine learning', 'deep learning', 'statistics', 
            'data visualization', 'tensorflow', 'pytorch', 'pandas', 'numpy', 
            'scikit-learn', 'hypothesis testing', 'a/b testing', 'big data'
        ],
        'Software Engineer': [
            'java', 'python', 'javascript', 'c++', 'algorithms', 'data structures', 
            'object-oriented programming', 'git', 'agile', 'testing', 'debugging', 
            'databases', 'api development', 'cloud computing', 'microservices'
        ],
        'UX Designer': [
            'user research', 'wireframing', 'prototyping', 'usability testing', 
            'figma', 'sketch', 'adobe xd', 'user flows', 'information architecture', 
            'visual design', 'interaction design', 'accessibility', 'html', 'css', 'design thinking'
        ],
        'Product Manager': [
            'product strategy', 'user stories', 'market research', 'roadmapping', 
            'agile', 'scrum', 'stakeholder management', 'analytics', 'a/b testing', 
            'presentation skills', 'leadership', 'communication', 'jira', 'project management',
            'competitive analysis'
        ],
        'DevOps Engineer': [
            'linux', 'aws', 'azure', 'docker', 'kubernetes', 'jenkins', 'ci/cd', 
            'infrastructure as code', 'terraform', 'ansible', 'monitoring', 'shell scripting', 
            'networking', 'security', 'python'
        ],
        'Data Analyst': [
            'sql', 'excel', 'tableau', 'power bi', 'data visualization', 'statistics',
            'python', 'r', 'business intelligence', 'data cleaning', 'data mining',
            'reporting', 'dashboards', 'forecasting', 'data modeling'
        ],
        'Frontend Developer': [
            'html', 'css', 'javascript', 'typescript', 'react', 'angular', 'vue',
            'responsive design', 'sass', 'webpack', 'redux', 'ui frameworks', 'jest',
            'accessibility', 'browser compatibility'
        ],
        'Backend Developer': [
            'java', 'python', 'node.js', 'c#', 'php', 'ruby', 'golang', 'rest apis',
            'graphql', 'databases', 'sql', 'nosql', 'microservices', 'authentication',
            'caching', 'security'
        ],
        'Network Engineer': [
            'cisco', 'networking', 'routing', 'switching', 'firewalls', 'vpn',
            'network security', 'tcpip', 'dns', 'dhcp', 'subnetting', 'wan',
            'lan', 'network monitoring', 'troubleshooting'
        ],
        'Cybersecurity Analyst': [
            'security', 'penetration testing', 'vulnerability assessment', 'ethical hacking',
            'firewall', 'incident response', 'security auditing', 'cryptography', 'risk management',
            'siem', 'security compliance', 'threat intelligence', 'security architecture'
        ]
    }
    
    # Experience levels and their corresponding years
    experience_levels = {
        'Entry': (0, 2),
        'Mid': (2, 5),
        'Senior': (5, 10),
        'Lead': (8, 15)
    }
    
    # Education levels
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education_weights = [0.05, 0.5, 0.35, 0.1]
    
    # Generate data
    data = []
    for _ in range(n_samples):
        # Select random job role
        job_role = random.choice(list(job_roles.keys()))
        
        # Select random skills for this role with increased noise
        role_skills = job_roles[job_role]
        # Core skills from the role - fewer core skills for more noise
        num_role_skills = random.randint(3, min(8, len(role_skills)))
        skills = random.sample(role_skills, num_role_skills)
        
        # Add more random skills from other roles (increased noise)
        other_skills = []
        for other_role in job_roles:
            if other_role != job_role:
                other_skills.extend(job_roles[other_role])
        
        # Increased noise - add up to 5 random skills
        num_other_skills = random.randint(1, 5)
        if num_other_skills > 0 and other_skills:
            skills.extend(random.sample(other_skills, min(num_other_skills, len(other_skills))))
        
        # Occasionally (5% chance) introduce a completely mismatched profile
        if random.random() < 0.05:
            wrong_job_role = random.choice([r for r in job_roles.keys() if r != job_role])
            job_role = wrong_job_role  # Deliberately mismatch skills and role
        
        # Convert skills list to a string representation
        skills_str = ', '.join(skills)
        
        # Select experience level and years with some correlation to job role
        if job_role in ['Data Scientist', 'DevOps Engineer']:
            exp_level_weights = [0.2, 0.35, 0.35, 0.1]  # More likely to be mid or senior
        elif job_role == 'Software Engineer':
            exp_level_weights = [0.25, 0.4, 0.25, 0.1]  # Good mix of all levels
        elif job_role == 'UX Designer':
            exp_level_weights = [0.25, 0.4, 0.25, 0.1]  # Good mix of all levels
        else:  # Product Manager
            exp_level_weights = [0.1, 0.3, 0.4, 0.2]  # More likely to be senior or lead
        
        exp_level = random.choices(list(experience_levels.keys()), weights=exp_level_weights)[0]
        min_years, max_years = experience_levels[exp_level]
        years_experience = round(random.uniform(min_years, max_years), 1)
        
        # Select education level with some correlation to job role
        if job_role == 'Data Scientist':
            edu_weights = [0.01, 0.3, 0.5, 0.19]  # More likely to have advanced degrees
        elif job_role == 'Software Engineer':
            edu_weights = [0.05, 0.6, 0.3, 0.05]  # More likely to have bachelor's
        elif job_role == 'UX Designer':
            edu_weights = [0.05, 0.65, 0.28, 0.02]  # More likely to have bachelor's
        elif job_role == 'DevOps Engineer':
            edu_weights = [0.08, 0.62, 0.28, 0.02]  # More likely to have bachelor's
        else:  # Product Manager
            edu_weights = [0.01, 0.55, 0.4, 0.04]  # More likely to have bachelor's or master's
        
        education = random.choices(education_levels, weights=edu_weights)[0]
        
        # Add whether they have certifications (boolean)
        has_certifications = random.choice([True, False])
        
        # Add data point
        data.append({
            'skills': skills_str,
            'years_experience': years_experience,
            'education': education,
            'has_certifications': has_certifications,
            'job_role': job_role
        })
    
    return pd.DataFrame(data)

# Generate and display dataset
df = generate_job_dataset(1000)
# print(f"Dataset shape: {df.shape}")
# print("\nFirst few rows:")
# print(df.head())
# print("\nClass distribution:")
# print(df['job_role'].value_counts())

# Feature engineering - process skills column
def preprocess_data(df):
    X = df.drop('job_role', axis=1)
    y = df['job_role']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create skills vocabulary from training data
    vectorizer = CountVectorizer(max_features=100)
    skills_train = vectorizer.fit_transform(X_train['skills']).toarray()
    skills_test = vectorizer.transform(X_test['skills']).toarray()
    
    # Create feature names for the vectorized skills
    skill_feature_names = [f'skill_{feat}' for feat in vectorizer.get_feature_names_out()]
    
    # Convert to DataFrames
    X_train_skills = pd.DataFrame(skills_train, columns=skill_feature_names)
    X_test_skills = pd.DataFrame(skills_test, columns=skill_feature_names)
    
    # Add other features
    X_train_skills['years_experience'] = X_train['years_experience'].values
    X_test_skills['years_experience'] = X_test['years_experience'].values
    
    # One-hot encode education
    edu_encoder = OneHotEncoder(sparse_output=False, drop='first')
    edu_train = edu_encoder.fit_transform(X_train[['education']])
    edu_test = edu_encoder.transform(X_test[['education']])
    
    edu_feature_names = [f'edu_{cat}' for cat in edu_encoder.categories_[0][1:]]
    X_train_skills[edu_feature_names] = edu_train
    X_test_skills[edu_feature_names] = edu_test
    
    # Add certification
    X_train_skills['has_certifications'] = X_train['has_certifications'].astype(int).values
    X_test_skills['has_certifications'] = X_test['has_certifications'].astype(int).values
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_skills['years_experience'] = scaler.fit_transform(X_train_skills[['years_experience']])
    X_test_skills['years_experience'] = scaler.transform(X_test_skills[['years_experience']])
    
    return X_train_skills, X_test_skills, y_train, y_test, vectorizer

X_train, X_test, y_train, y_test, vectorizer = preprocess_data(df)

# print("\nPreprocessed features shape:")
# print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
# print(f"Top skills in vocabulary: {vectorizer.get_feature_names_out()[:10]}")

# # Visualize correlation between features and target
# plt.figure(figsize=(12, 8))
# sns.countplot(data=df, x='job_role')
# plt.title('Distribution of Job Roles')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('job_role_distribution.png')

# Visualize distribution of experience by role
# plt.figure(figsize=(12, 8))
# sns.boxplot(data=df, x='job_role', y='years_experience')
# plt.title('Years of Experience by Job Role')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig('experience_by_role.png')

# Show education distribution
# plt.figure(figsize=(12, 8))
df_edu = pd.crosstab(df['job_role'], df['education'])
df_edu_pct = df_edu.div(df_edu.sum(axis=1), axis=0)
df_edu_pct.plot(kind='bar', stacked=True, figsize=(12, 8))
# plt.title('Education Distribution by Job Role')
# plt.xlabel('Job Role')
# plt.ylabel('Percentage')
# plt.xticks(rotation=45, ha='right')
# plt.legend(title='Education Level')
# plt.tight_layout()
# plt.savefig('education_by_role.png')

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Define models
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        # print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix
        }
        
        # Print results
        # print(f"{name} Accuracy: {accuracy:.4f}")
        # print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Plot confusion matrix
        # plt.figure(figsize=(8, 6))
        # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
        #            xticklabels=model.classes_, yticklabels=model.classes_)
        # plt.title(f'Confusion Matrix - {name}')
        # plt.ylabel('True Label')
        # plt.xlabel('Predicted Label')
        # plt.tight_layout()
        # plt.savefig(f'confusion_matrix_{name.replace(" ", "_").lower()}.png')
    
    return results

# Train and evaluate all models
results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

# Compare models
accuracies = [results[model]['accuracy'] for model in results]
model_names = list(results.keys())

# plt.figure(figsize=(10, 6))
# bars = plt.bar(model_names, accuracies)
# plt.title('Model Comparison - Accuracy')
# plt.xlabel('Model')
# plt.ylabel('Accuracy')
# plt.ylim(0.7, 1.0)  # Set y-axis to start from 0.7 for better visualization
# plt.xticks(rotation=45, ha='right')

# Add accuracy values on top of bars
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
#             f'{height:.4f}', ha='center', va='bottom')

# plt.tight_layout()
# plt.savefig('model_comparison.png')

# Feature importance analysis (for tree-based models)
# def plot_feature_importance(model, feature_names, title):
#     if hasattr(model, 'feature_importances_'):
#         importances = model.feature_importances_
#         indices = np.argsort(importances)[-20:]  # Get indices of top 20 features
        
#         plt.figure(figsize=(12, 8))
#         plt.title(title)
#         plt.barh(range(len(indices)), importances[indices], align='center')
#         plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#         plt.xlabel('Feature Importance')
#         plt.tight_layout()
#         plt.savefig(f'{title.replace(" ", "_").lower()}.png')

# # Plot feature importance for Decision Tree and Random Forest
# plot_feature_importance(results['Decision Tree']['model'], X_train.columns, 'Decision Tree Feature Importance')
# plot_feature_importance(results['Random Forest']['model'], X_train.columns, 'Random Forest Feature Importance')

# Create a function to make predictions for new data
def predict_job_role(skills_text, years_experience, education, has_certifications, vectorizer, model):
    # Process skills
    skills_vector = vectorizer.transform([skills_text]).toarray()
    
    # Create feature names for the vectorized skills
    skill_feature_names = [f'skill_{feat}' for feat in vectorizer.get_feature_names_out()]
    
    # Convert to DataFrame
    features = pd.DataFrame(skills_vector, columns=skill_feature_names)
    
    # Add other features (assuming scaler and encoders are available)
    features['years_experience'] = years_experience
    
    # Make sure all education columns from training data are present
    edu_columns = [col for col in X_train.columns if col.startswith('edu_')]
    for col in edu_columns:
        features[col] = 0
    
    # Set the appropriate education column to 1 (if it exists)
    edu_col = f'edu_{education}'
    if edu_col in features.columns:
        features[edu_col] = 1
    
    # Add certification
    features['has_certifications'] = int(has_certifications)
    
    # Make sure all columns from training are present and in the same order
    missing_columns = set(X_train.columns) - set(features.columns)
    for col in missing_columns:
        features[col] = 0
    
    # Ensure columns are in the same order as during training
    features = features[X_train.columns]
    
    # Make prediction
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)
    
    # Get top 2 most likely job roles with probabilities
    top_indices = np.argsort(probabilities[0])[-2:][::-1]
    top_roles = [(model.classes_[idx], probabilities[0][idx]) for idx in top_indices]
    
    return prediction[0], top_roles

# Example prediction using the Random Forest model (considered best performing)
best_model = results['Random Forest']['model']
new_skills = "python, pandas, sql, data visualization, statistics"
predicted_role, top_roles = predict_job_role(new_skills, 3, "Master", True, vectorizer, best_model)

# print("\nExample Prediction:")
# print(f"Skills: {new_skills}")
# print(f"Years of Experience: 3, Education: Master, Has Certifications: True")
# print(f"Predicted Job Role: {predicted_role}")
# print("Top 2 most likely roles:")
# for role, prob in top_roles:
#     print(f"- {role}: {prob:.4f} ({prob*100:.1f}%)")

# Summary of model performance metrics
# print("\nModel Performance Summary:")
# summary = {
#     'Model': [],
#     'Accuracy': [],
#     'Macro Avg F1-Score': [],
#     'Weighted Avg F1-Score': []
# }

# for name, result in results.items():
#     summary['Model'].append(name)
#     summary['Accuracy'].append(result['accuracy'])
#     summary['Macro Avg F1-Score'].append(result['classification_report']['macro avg']['f1-score'])
#     summary['Weighted Avg F1-Score'].append(result['classification_report']['weighted avg']['f1-score'])

# summary_df = pd.DataFrame(summary)
# print(summary_df.to_string(index=False))

# Save results to CSV
# summary_df.to_csv('model_performance_summary.csv', index=False)



def predict_job_role(skills, years_experience, education, has_certifications, vectorizer=vectorizer, model=results['Random Forest']['model']):
    # Vectorize skills
    skills_vec = vectorizer.transform([skills]).toarray()
    skills_df = pd.DataFrame(skills_vec, columns=[f'skill_{feat}' for feat in vectorizer.get_feature_names_out()])
    
    # Add other features
    skills_df['years_experience'] = StandardScaler().fit_transform([[years_experience]])[0][0]
    
    # One-hot encode education (assuming same encoder and categories as training)
    edu_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    for level in edu_levels[1:]:  # dropped first during training
        skills_df[f'edu_{level}'] = int(education == level)
    
    # Add certifications
    skills_df['has_certifications'] = int(has_certifications)
    
    # Fill missing skill columns if any (to match training features)
    for col in X_train.columns:
        if col not in skills_df.columns:
            skills_df[col] = 0
    skills_df = skills_df[X_train.columns]  # ensure order matches
    
    # Predict
    prediction = model.predict(skills_df)
    return prediction[0]

# Sample 1: Software Engineer-like profile
# predict_job_role(
#     skills="Python, SQL",
#     years_experience=3.5,
#     education="Bachelor",
#     has_certifications=True,
#     vectorizer=vectorizer,
#     model=results['Decision Tree']['model']
# )

# Sample 2: Data Scientist-like profile
# predict_job_role(
#     skills="python, pandas, numpy, machine learning, tensorflow",
#     years_experience=4.2,
#     education="Master",
#     has_certifications=False,
#     vectorizer=vectorizer,
#     model=results['Random Forest']['model']
# )

# # Sample 3: UX Designer-like profile
# predict_job_role(
#     skills="figma, wireframing, user flows, visual design, prototyping",
#     years_experience=2.0,
#     education="Bachelor",
#     has_certifications=False,
#     vectorizer=vectorizer,
#     model=results['Random Forest']['model']
# )

# # Sample 4: DevOps Engineer-like profile
# predict_job_role(
#     skills="docker, kubernetes, aws, ci/cd, linux",
#     years_experience=6.5,
#     education="Bachelor",
#     has_certifications=True,
#     vectorizer=vectorizer,
#     model=results['Random Forest']['model']
# )
