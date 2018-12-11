from django.shortcuts import render
from django.http import HttpResponse
import json
from learn import wiki


# Create your views here.
def hello(request):
    return HttpResponse('helloWord')


def home(request):
    return render(request, 'index.html')


def search(request):
    ctx = {}
    if request.POST:
        print(request.POST)
        if 'delete' in request.POST:
            dels = request.POST['delete']
        else:
            dels = 'false'
        wiki.delfile(dels)
        if 'q' in request.POST:
            s = request.POST['q']
            nodes, rels = wiki.get_triples(s)
            ctx['nodes'] = json.dumps(nodes)
            ctx['rels'] = json.dumps(rels)
            ctx['content'] = s
        if 'run' in request.POST:
            runs = request.POST['run']
            nodes, rels = wiki.runfile(runs)
            ctx['nodes'] = json.dumps(nodes)
            ctx['rels'] = json.dumps(rels)
    return render(request, "index.html", ctx)

# def search_name(request):
#     ctx = {}
#     if request.POST:
#         selects = ['name', 'screen_name']
#         s = request.POST['q']
#         a = request.POST['mod']
#         head, data = gd.search(s, mod=a)
#         ctx['head'] = json.dumps(head)
#         ctx['data'] = json.dumps(data)
#         ctx['content'] = s
#         ctx['mode'] = a
#         ctx['ind'] = selects.index(a)
#
#     return render(request, 'search.html', ctx)
