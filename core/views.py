from django.shortcuts import render
from .forms import JobPredictionForm
from .predictor import predict_job_role  # Your existing prediction function

def predict_role_view(request):
    result = None
    if request.method == 'POST':
        form = JobPredictionForm(request.POST)
        if form.is_valid():
            # Process form data
            skills = ', '.join(form.cleaned_data['skills'])
            years = form.cleaned_data['years_experience']
            education = form.cleaned_data['education']
            certifications = form.cleaned_data['has_certifications']
            
            # Get prediction
            result = predict_job_role(
                skills=skills,
                years_experience=years,
                education=education,
                has_certifications=certifications
            )
            print(f"users predicted job role: {result}")
    else:
        form = JobPredictionForm()
    
    return render(request, 'predictor.html', {
        'form': form,
        'result': result
    })