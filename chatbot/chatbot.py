"""
chatbot.py — AI chatbot powered by OpenRouter + ML prediction engine
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from models.predict import predict_property, top_properties

# Load .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# ── Market summary ─────────────────────────────────────────
def build_market_summary():
    try:
        df = pd.read_csv("data/processed/listings_featured.csv")
        lines = ["=== LIVE MARKET DATA SUMMARY ==="]
        for city, grp in df.groupby("city"):
            lines.append(f"\n{city}:")
            lines.append(f"  Avg nightly price: ${grp['nightly_price_usd'].mean():.0f}")
            lines.append(f"  Avg occupancy: {grp['occupancy_rate'].mean():.0%}")
            lines.append(f"  Avg ROI: {grp['roi_pct'].mean():.1f}%")
            lines.append(f"  Best area by ROI: {grp.groupby('area')['roi_pct'].mean().idxmax()}")
            lines.append(f"  Total listings: {len(grp)}")
        return "\n".join(lines)
    except Exception as e:
        return f"Market data unavailable: {e}"

# ── Top listings ─────────────────────────────────────────
def build_top_listings_context(city=None, top_n=5):
    try:
        df = top_properties(city=city, min_roi=5.0, top_n=top_n)
        lines = [f"\n=== TOP {top_n} PROPERTIES {'in '+city if city else 'OVERALL'} ==="]
        for _, row in df.iterrows():
            lines.append(
                f"{row['city']} - {row['area']} | {row['property_type']} | "
                f"{row['bedrooms']}BR | ${row['nightly_price_usd']}/night | "
                f"ROI: {row['roi_pct']}% | Score: {row['investment_score']}/100"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Listings unavailable: {e}"

SYSTEM_PROMPT = """You are PropPredict AI — an expert AI assistant for Airbnb investment.

Capabilities:
- Predict Airbnb price and ROI
- Recommend best investment areas
- Compare cities (Dubai, Goa, Maldives)

Always:
- Use bullet points
- Be concise (<150 words)
- Include numbers

{market_summary}

{top_listings}
"""

class PropertyChatbot:
    def __init__(self):
        self.history = []
        self.market_summary = build_market_summary()
        self.top_listings = build_top_listings_context()

    def detect_city(self, text):
        t = text.lower()
        if "dubai" in t: return "Dubai"
        if "goa" in t: return "Goa"
        if "maldives" in t: return "Maldives"
        return None

    def extract_property_params(self, text):
        params = {}
        t = text.lower()

        city = self.detect_city(text)
        if city:
            params["city"] = city

        if "villa" in t: params["property_type"] = "Villa"
        if "apartment" in t: params["property_type"] = "Apartment"

        if "1" in t: params["bedrooms"] = 1
        if "2" in t: params["bedrooms"] = 2
        if "3" in t: params["bedrooms"] = 3

        return params

    def chat(self, user_message):
        city_hint = self.detect_city(user_message)

        # ML prediction
        enrichment = ""
        try:
            params = self.extract_property_params(user_message)
            if params.get("city"):
                result = predict_property(params)

                enrichment = f"""
ML OUTPUT:
Price: ${result['predicted_nightly_price_usd']}
ROI: {result['estimated_roi_pct']}%
Score: {result['investment_score']}
"""
        except:
            pass

        full_prompt = user_message + "\n" + enrichment

        self.history.append({"role": "user", "content": full_prompt})

        if not OPENROUTER_API_KEY:
            return "⚠️ No API key found. Add OPENROUTER_API_KEY in .env"

        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-3-8b-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT.format(
                        market_summary=self.market_summary,
                        top_listings=self.top_listings
                    )},
                    *self.history
                ],
                max_tokens=300
            )

            reply = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": reply})
            return reply

        except Exception as e:
            return f"API Error: {e}"

    def reset(self):
        self.history = []

if __name__ == "__main__":
    bot = PropertyChatbot()
    print("PropPredict AI (OpenRouter) — type 'quit'\n")

    while True:
        q = input("You: ")
        if q.lower() in ["quit", "exit"]:
            break
        print("\nBot:", bot.chat(q), "\n")