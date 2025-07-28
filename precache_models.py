import sys
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline
    print('Pre-caching SentenceTransformer...')
    SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    print('Pre-caching Transformers pipeline...')
    pipeline('text2text-generation', model='google/flan-t5-small', device=-1)
    print('Models pre-cached successfully')
except Exception as e:
    print(f'Error pre-caching models: {e}')
    sys.exit(1)