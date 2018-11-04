# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

from maria.master.ontology_classifier import OntologyPredictor

def predict1(request):
    op = OntologyPredictor()
    ontology=request.GET.get('ontology','')
    title = request.GET.get('title', '')
    gender=  request.GET.get('gender', '')
    input = {}
    input['title'] = title
    input['ontology'] = ontology
    input['gender'] = gender
    predicted_ontologies = {}
    predicted = op.predict(input, use_title_first=True, title_override=True)
    predicted_ontologies['ontology'] = predicted
    return JsonResponse(predicted_ontologies)
# Create your views here.
