import pandas as pd
import numpy as np
import os, random

random.seed(42)
np.random.seed(42)

# ── Dubai ─────────────────────────────────────────────────────────────────────
dubai_areas = ["Dubai Marina","JVC","Palm Jumeirah","Downtown Dubai",
               "Business Bay","JBR","Dubai Hills","Silicon Oasis","Deira","DIFC"]
dubai_props = ["Apartment","Villa","Studio","Penthouse","Townhouse"]
dubai_amen  = ["Pool","Gym","Parking","WiFi","AC","Washer","Balcony","Sea View","Kitchen","Doorman"]
area_mult_dubai = {"Palm Jumeirah":2.2,"DIFC":1.8,"Downtown Dubai":1.7,"Dubai Marina":1.5,
                   "JBR":1.4,"Business Bay":1.3,"JVC":1.0,"Dubai Hills":1.2,
                   "Silicon Oasis":0.8,"Deira":0.75}
base_price_dubai = {"Studio":80,"Apartment":120,"Villa":300,"Penthouse":400,"Townhouse":200}

def make_dubai(i):
    area  = random.choice(dubai_areas)
    ptype = random.choice(dubai_props)
    beds  = random.randint(0, 4)
    baths = max(1, beds + random.randint(-1, 1))
    price = round(base_price_dubai[ptype] * area_mult_dubai[area] * (1 + beds*0.25) * np.random.uniform(0.85,1.15))
    rating = round(np.random.uniform(3.8, 5.0), 2)
    occ    = round(np.random.uniform(0.45, 0.92), 2)
    dist_beach = round(np.random.uniform(0.1, 8.0), 1)
    amenities  = "|".join(random.sample(dubai_amen, random.randint(4,8)))
    reviews    = random.randint(5, 420)
    annual_rev = round(price * occ * 365 * 0.72)
    prop_price = round(annual_rev / 0.08 * np.random.uniform(0.85,1.15))
    roi = round((annual_rev / prop_price)*100, 2)
    invest_score = min(100, round(roi*5 + rating*8 + occ*20 + (1/max(dist_beach,0.1))*2))
    return {"id":f"DXB{i:04d}","city":"Dubai","area":area,"property_type":ptype,
            "bedrooms":beds,"bathrooms":baths,"nightly_price_usd":price,
            "review_score":rating,"occupancy_rate":occ,
            "distance_to_beach_km":dist_beach,"distance_to_airport_km":round(np.random.uniform(5,35),1),
            "num_reviews":reviews,"amenities":amenities,"annual_revenue_usd":annual_rev,
            "property_price_usd":prop_price,"roi_pct":roi,"investment_score":invest_score,
            "latitude":round(25.0 + np.random.uniform(-0.3,0.3),4),
            "longitude":round(55.1 + np.random.uniform(-0.2,0.4),4)}

# ── Goa ───────────────────────────────────────────────────────────────────────
goa_areas = ["Calangute","Anjuna","Vagator","Candolim","Baga","Palolem","Arambol","Morjim","Colva","Panaji"]
goa_props = ["Villa","Cottage","Apartment","Beach Hut","Homestay"]
goa_amen  = ["Pool","Garden","Parking","WiFi","AC","Hammock","Outdoor Shower","Beach Access","BBQ","Yoga Deck"]
area_mult_goa = {"Vagator":1.5,"Anjuna":1.4,"Morjim":1.4,"Calangute":1.2,"Arambol":1.3,
                 "Candolim":1.1,"Baga":1.0,"Palolem":1.0,"Colva":0.9,"Panaji":0.85}
base_price_goa = {"Villa":90,"Cottage":40,"Apartment":35,"Beach Hut":25,"Homestay":20}

def make_goa(i):
    area  = random.choice(goa_areas)
    ptype = random.choice(goa_props)
    beds  = random.randint(1, 5)
    baths = max(1, beds + random.randint(-1,1))
    price = round(base_price_goa[ptype] * area_mult_goa[area] * (1 + beds*0.3) * np.random.uniform(0.8,1.2))
    rating = round(np.random.uniform(3.5, 5.0), 2)
    occ_peak = round(np.random.uniform(0.70, 0.95), 2)
    occ_off  = round(np.random.uniform(0.20, 0.50), 2)
    occ_avg  = round((occ_peak*5 + occ_off*7)/12, 2)
    dist_beach = round(np.random.uniform(0.05, 3.0), 1)
    amenities  = "|".join(random.sample(goa_amen, random.randint(3,7)))
    reviews    = random.randint(3, 280)
    annual_rev = round(price * occ_avg * 365 * 0.70)
    prop_price = round(annual_rev / 0.09 * np.random.uniform(0.9,1.1))
    roi = round((annual_rev / prop_price)*100, 2)
    invest_score = min(100, round(roi*5 + rating*8 + occ_avg*20 + (1/max(dist_beach,0.1))*3))
    return {"id":f"GOA{i:04d}","city":"Goa","area":area,"property_type":ptype,
            "bedrooms":beds,"bathrooms":baths,"nightly_price_usd":price,
            "review_score":rating,"occupancy_rate":occ_avg,
            "peak_occupancy":occ_peak,"off_occupancy":occ_off,
            "distance_to_beach_km":dist_beach,"distance_to_airport_km":round(np.random.uniform(5,45),1),
            "num_reviews":reviews,"amenities":amenities,"annual_revenue_usd":annual_rev,
            "property_price_usd":prop_price,"roi_pct":roi,"investment_score":invest_score,
            "latitude":round(15.3 + np.random.uniform(-0.4,0.6),4),
            "longitude":round(73.9 + np.random.uniform(-0.2,0.2),4)}

# ── Maldives ──────────────────────────────────────────────────────────────────
maldives_areas = ["North Male Atoll","South Male Atoll","Baa Atoll","Ari Atoll",
                  "Lhaviyani Atoll","Noonu Atoll","Addu Atoll","Vaavu Atoll"]
maldives_props = ["Overwater Bungalow","Beach Villa","Water Villa","Garden Room","Sunset Villa"]
maldives_amen  = ["Private Pool","Direct Lagoon Access","Butler Service","Snorkeling Gear",
                  "WiFi","AC","Outdoor Shower","Glass Floor","Sun Deck","Kayak"]
area_mult_maldives = {"North Male Atoll":1.5,"Baa Atoll":1.4,"Ari Atoll":1.3,"Lhaviyani Atoll":1.2,
                      "Noonu Atoll":1.1,"Vaavu Atoll":1.0,"South Male Atoll":1.2,"Addu Atoll":0.9}
base_price_maldives = {"Overwater Bungalow":500,"Beach Villa":350,"Water Villa":600,
                       "Garden Room":200,"Sunset Villa":450}

def make_maldives(i):
    area  = random.choice(maldives_areas)
    ptype = random.choice(maldives_props)
    beds  = random.randint(1, 3)
    baths = beds
    price = round(base_price_maldives[ptype] * area_mult_maldives[area] * np.random.uniform(0.85,1.15))
    rating = round(np.random.uniform(4.2, 5.0), 2)
    occ    = round(np.random.uniform(0.55, 0.90), 2)
    dist_airport = round(np.random.uniform(0.2, 3.5), 1)
    amenities = "|".join(random.sample(maldives_amen, random.randint(5,9)))
    reviews   = random.randint(10, 600)
    annual_rev = round(price * occ * 365 * 0.75)
    prop_price = round(annual_rev / 0.06 * np.random.uniform(0.9,1.1))
    roi = round((annual_rev / prop_price)*100, 2)
    invest_score = min(100, round(roi*6 + rating*9 + occ*18))
    return {"id":f"MLD{i:04d}","city":"Maldives","area":area,"property_type":ptype,
            "bedrooms":beds,"bathrooms":baths,"nightly_price_usd":price,
            "review_score":rating,"occupancy_rate":occ,
            "distance_to_beach_km":0.0,"distance_to_airport_km":dist_airport,
            "num_reviews":reviews,"amenities":amenities,"annual_revenue_usd":annual_rev,
            "property_price_usd":prop_price,"roi_pct":roi,"investment_score":invest_score,
            "latitude":round(4.0 + np.random.uniform(-2.0,5.0),4),
            "longitude":round(73.5 + np.random.uniform(-1.0,3.0),4)}

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    dubai_df    = pd.DataFrame([make_dubai(i)    for i in range(1,601)])
    goa_df      = pd.DataFrame([make_goa(i)      for i in range(1,501)])
    maldives_df = pd.DataFrame([make_maldives(i) for i in range(1,401)])
    combined    = pd.concat([dubai_df, goa_df, maldives_df], ignore_index=True)
    combined["peak_occupancy"] = combined["peak_occupancy"].fillna(combined["occupancy_rate"])
    combined["off_occupancy"]  = combined["off_occupancy"].fillna(combined["occupancy_rate"]*0.5)

    dubai_df.to_csv("data/raw/dubai_listings.csv", index=False)
    goa_df.to_csv("data/raw/goa_listings.csv", index=False)
    maldives_df.to_csv("data/raw/maldives_listings.csv", index=False)
    combined.to_csv("data/raw/combined_listings.csv", index=False)
    print(f"Dubai: {len(dubai_df)} | Goa: {len(goa_df)} | Maldives: {len(maldives_df)} | Total: {len(combined)}")
    print(combined.groupby("city")[["nightly_price_usd","roi_pct","occupancy_rate"]].mean().round(2))
