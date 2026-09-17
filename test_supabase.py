
from supabase import create_client

url = "https://svzrazwfbojkdshpudck.supabase.co"
key = "YOUR_SECRET_KEY"

supabase = create_client(url, key)

print("Connected OK")

print(
    supabase.storage.list_buckets()
)
