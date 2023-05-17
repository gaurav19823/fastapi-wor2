from fastapi import FastAPI, Form, Request
from gensim.models.keyedvectors import KeyedVectors
from fastapi.middleware.cors import CORSMiddleware

model_path = "word2vec_2lac_20epo.bin"
model = KeyedVectors.load_word2vec_format(model_path, binary=True)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/similar")
async def get_similar_words(request: Request, word: str = Form(...)):
    similar_words = model.most_similar(word, topn=5)
    return {"similar_words": similar_words}
#uvicorn app:app --reload