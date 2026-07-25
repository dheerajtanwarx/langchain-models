# Ye project ka simple goal hai:
# User ek query likhega, aur program documents me se meaning ke basis par sabse related document find karega.

from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# embedding model load hota h Ye model text ko numbers/vector me convert karta hai. for ex: "Zomato is a food delivery platform"=>[0.12, -0.54, 0.88, ...]
embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Ye tumhara small database hai. Real projects me ye PDFs, articles, product descriptions, FAQs, resumes, etc. ho sakte hain.
documents = [
    "Zomato is a food delivery platform that lets users order meals from nearby restaurants.",
    "Swiggy allows customers to order food online and get it delivered to their home.",
    "Restaurants use food delivery apps to reach more customers and manage online orders.",
    "A pizza shop can receive online orders through delivery platforms and send food to customers.",
    "Online payment systems allow users to pay for food orders using cards, UPI, or wallets.",
    "Delivery partners pick up meals from restaurants and deliver them to customers.",
    "A restaurant menu contains food items, prices, descriptions, and available offers.",
    "Customer reviews help people choose good restaurants based on food quality and service.",
    "Google Maps helps users find nearby restaurants, cafes, and food places.",
    "Cloud kitchens prepare food only for online delivery and usually do not offer dine-in service."
]

# query = "who brings the food to the customer"
# query = "what is zomato"
# query = "How is zomato works"
query = "Whcih is the food delivery platform"


# Har document numbers/vector me convert ho jata hai. Agar 10 documents hain, to 10 vectors banenge.
doc_embeddings = embedding.embed_documents(documents)
# har query vector me convert ho jaye gi or yhi vector compare hoga documents vector se or hume answer mile ga 
query_embedding = embedding.embed_query(query)

# cosine_similarity(...) kya karta hai? Ye query vector ko saare document vectors se compare karta hai.  But cosine_similarity ko usually 2D list chahiye hoti hai. Isliye hum isko extra brackets me rakhte hain: [query_embedding] kyuki embed_query hume 1d vector deta h 
score = cosine_similarity([query_embedding], doc_embeddings)[0]

# scores ko index ke saath jodo, score ke basis par sort karo, last/highest score wala uthao,uska index aur score alag variables me store karo.
index, score = (sorted(list(enumerate(score)),key=lambda x:x[1])[-1])

print(query)
print(documents[index])
print("similarity score is: ", score)