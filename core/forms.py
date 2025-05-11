from django import forms

# Static job skills data (put this in utils.py in real projects)
job_skills = {
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

# Get all unique skills
all_skills = sorted({skill for skills in job_skills.values() for skill in skills})
SKILL_CHOICES = [(skill, skill.title()) for skill in all_skills]

EDUCATION_CHOICES = [
    ('Bachelor', "Bachelor's Degree"),
    ('Master', "Master's Degree"),
    ('PhD', "PhD"),
    ('Other', "Other"),
]

class JobPredictionForm(forms.Form):
    skills = forms.MultipleChoiceField(
        choices=SKILL_CHOICES,
        widget=forms.CheckboxSelectMultiple,
        required=True
    )
    years_experience = forms.FloatField(
        min_value=0,
        label="Years of Experience"
    )
    education = forms.ChoiceField(
        choices=EDUCATION_CHOICES,
        widget=forms.RadioSelect
    )
    has_certifications = forms.BooleanField(
        required=False,
        label="Do you have professional certifications?"
    )