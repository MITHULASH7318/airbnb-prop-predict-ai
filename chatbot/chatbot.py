"""
chatbot.py — AI chatbot powered by Claude API + ML prediction engine
No LangChain needed — we pass property context directly in the system prompt
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import requests
from models.predict import predict_property, top_properties

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── Build market context summary (injected into every prompt) ─────────────────
def build_market_summary() -> str:
    try:
        # Resolve path for both local and Streamlit Cloud
        for base in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]:
            p = os.path.join(base, "data", "processed", "listings_featured.csv")
            if os.path.exists(p):
                df = pd.read_csv(p)
                break
        else:
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

# ── Top properties context ─────────────────────────────────────────────────────
def build_top_listings_context(city=None, top_n=5) -> str:
    try:
        df = top_properties(city=city, min_roi=5.0, top_n=top_n)
        lines = [f"\n=== TOP {top_n} PROPERTIES {'in '+city if city else 'OVERALL'} ==="]
        for _, row in df.iterrows():
            lines.append(
                f"  {row['id']} | {row['city']} - {row['area']} | {row['property_type']} "
                f"| {row['bedrooms']}BR | ${row['nightly_price_usd']}/night "
                f"| ROI: {row['roi_pct']}% | Score: {row['investment_score']}/100 "
                f"| Signal: {row['investment_label']}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Listings unavailable: {e}"

SYSTEM_PROMPT = """You are PropPredict AI — an expert AI assistant for Airbnb property investment 
across Dubai, Goa, and Maldives. You have access to real market data and ML model predictions.

Your capabilities:
1. Predict nightly rental prices for any property configuration
2. Estimate ROI and investment score (0-100)  
3. Recommend best areas to buy in each city
4. Compare properties across Dubai, Goa, and Maldives
5. Explain what drives Airbnb prices (beach distance, amenities, seasonality etc.)
6. Answer questions about Airbnb regulations, taxes, property buying rules

Tone: Professional but friendly. Always back answers with numbers from the data.
Format: Use bullet points for lists. Bold key numbers. Keep responses under 200 words.
When user asks for prediction, extract property details and tell them you are running the ML model.

{market_summary}

{top_listings}
"""

class PropertyChatbot:
    def __init__(self):
        self.history = []
        self.market_summary = build_market_summary()
        self.top_listings   = build_top_listings_context()

    def get_system_prompt(self, city_hint=None):
        top = build_top_listings_context(city_hint) if city_hint else self.top_listings
        return SYSTEM_PROMPT.format(
            market_summary=self.market_summary,
            top_listings=top
        )

    def detect_city(self, text: str):
        text_lower = text.lower()
        if "dubai"    in text_lower: return "Dubai"
        if "goa"      in text_lower: return "Goa"
        if "maldives" in text_lower: return "Maldives"
        return None

    def extract_property_params(self, text: str) -> dict:
        params = {}
        t = text.lower()
        # City
        city = self.detect_city(text)
        if city: params["city"] = city
        # Bedrooms
        for n in range(6):
            if f"{n} bed" in t or f"{n}br" in t or f"{n}-bed" in t:
                params["bedrooms"] = n
        # Property type
        for pt in ["villa","apartment","studio","penthouse","cottage","bungalow","resort"]:
            if pt in t:
                params["property_type"] = pt.title()
        # Budget
        import re
        budget = re.findall(r'\$?([\d,]+)k?', t)
        if budget:
            val = int(budget[0].replace(",",""))
            if "k" in t[t.find(budget[0]):t.find(budget[0])+6]:
                val *= 1000
            if val > 1000:
                params["property_price_hint"] = val
        return params

    def chat(self, user_message: str) -> str:
        """Send message, optionally enrich with ML predictions, return response."""
        city_hint = self.detect_city(user_message)

        # Check if user wants a prediction
        prediction_keywords = ["predict","price","how much","roi","worth","invest",
                                "buy","return","score","nightly","estimate"]
        wants_prediction = any(kw in user_message.lower() for kw in prediction_keywords)

        enrichment = ""
        if wants_prediction:
            params = self.extract_property_params(user_message)
            if params.get("city"):
                try:
                    result = predict_property(params)
                    enrichment = f"""
[ML MODEL OUTPUT — use these numbers in your response]:
Predicted nightly price: ${result['predicted_nightly_price_usd']}
Predicted occupancy: {result['predicted_occupancy_rate']:.0%}
Predicted annual revenue: ${result['predicted_annual_revenue_usd']:,}
Estimated property price: ${result['estimated_property_price_usd']:,}
Estimated ROI: {result['estimated_roi_pct']}%
Investment signal: {result['investment_label']}
Investment score: {result['investment_score']}/100
"""
                except Exception as e:
                    enrichment = f"[ML model error: {e}]"

        full_message = user_message
        if enrichment:
            full_message = user_message + "\n\n" + enrichment

        self.history.append({"role": "user", "content": full_message})

        if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "":
            # Fallback: rule-based responses for demo without API key
            self.history.append({"role":"assistant","content":"[API key not set — see README]"})
            return self._fallback_response(user_message, enrichment)

        headers = {"Content-Type":"application/json","x-api-key":ANTHROPIC_API_KEY,
                   "anthropic-version":"2023-06-01"}
        body    = {"model":"claude-sonnet-4-20250514","max_tokens":600,
                   "system": self.get_system_prompt(city_hint),
                   "messages": self.history}
        try:
            resp = requests.post("https://api.anthropic.com/v1/messages",
                                 headers=headers, json=body, timeout=30)
            resp.raise_for_status()
            data    = resp.json()
            reply   = data["content"][0]["text"]
            self.history.append({"role":"assistant","content":reply})
            return reply
        except Exception as e:
            return f"API error: {e}. Make sure ANTHROPIC_API_KEY is set."

    def _fallback_response(self, msg: str, enrichment: str) -> str:
        """Rule-based fallback when no API key — for demo/testing."""
        msg_l = msg.lower()
        if enrichment:
            lines = [l for l in enrichment.split("\n") if l.strip() and "ML MODEL" not in l and "[" not in l]
            return ("Based on my ML model analysis:\n\n" + "\n".join(lines) +
                    "\n\n*Set ANTHROPIC_API_KEY for full AI responses.*")
        if "dubai" in msg_l:
            return ("Dubai top zones by ROI: JVC (7-9%), Silicon Oasis (8-10%), Dubai Hills (7-8%).\n"
                    "Avg nightly price: $453. Best property type: Apartments near metro.\n"
                    "*Set ANTHROPIC_API_KEY for detailed AI analysis.*")
        if "goa" in msg_l:
            return ("Goa best zones: Vagator & Anjuna (premium), Morjim (emerging).\n"
                    "Peak season Nov-Feb: 70-95% occupancy. Avg ROI: 9%.\n"
                    "*Set ANTHROPIC_API_KEY for detailed AI analysis.*")
        if "maldives" in msg_l:
            return ("Maldives top zone: North Malé Atoll. Overwater villas avg $750/night.\n"
                    "Avg occupancy 73%. Best ROI: Baa Atoll properties near marine reserves.\n"
                    "*Set ANTHROPIC_API_KEY for detailed AI analysis.*")
        return ("I can answer questions about Dubai, Goa, and Maldives properties!\n"
                "Try: 'Best area in Dubai under $500K?' or 'Predict price for 2BR villa in Goa'\n"
                "*Set ANTHROPIC_API_KEY for full AI-powered responses.*")

    def reset(self):
        self.history = []

if __name__ == "__main__":
    bot = PropertyChatbot()
    print("PropPredict AI — type 'quit' to exit\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit","exit","q"): break
        print(f"\nBot: {bot.chat(q)}\n")
