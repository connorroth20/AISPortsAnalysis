from django.shortcuts import render
from django.http import HttpResponse

# Create your views here (where we define view functions).
# View function takes in request and returns a response
# Request handler: request -> response
# Action = View
# In a function, you're able to do differnt things such as:
# Pull data from DB
# Transform data
# Send an email, etc

def say_hello(request):
    return render(request, 'hello.html', { 'name': 'Dan'})

# After creating this view function, we need to map it to a URL
# So when we get a request at that URL, this function will be caught
