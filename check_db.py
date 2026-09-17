from supabase import create_client, Client

# Your existing credentials
SUPABASE_URL = "https://svzrazwfbojkdshpudck.supabase.co"
SUPABASE_KEY = "YOUR_KEY"

def verify_connection():
    print("--- Starting Connection Check ---")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # 1. Test Database Connection
        # This attempts to read one row from your 'assessments' table
        db_test = supabase.table("patients").select("*").limit(1).execute()
        print("✅ Success: Database is reachable!")
        print(f"   Current data in table: {db_test.data}")

        # 2. Test Storage Bucket Connection
        # This attempts to list files in your 'mri-scans' bucket
        storage_test = supabase.storage.from_("mri-scans").list()
        print("✅ Success: Storage bucket 'mri-scans' is reachable!")
        print(f"   Files found in bucket: {len(storage_test)} files.")

    except Exception as e:
        print("❌ Connection Failed!")
        print(f"   Error Details: {str(e)}")

if __name__ == "__main__":
    verify_connection()