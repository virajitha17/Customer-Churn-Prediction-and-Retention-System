def assign_retention_strategy(segments):
    def strategy(segment):
        if segment == "High Risk":
            return "Offer Discount or Loyalty Program"
        else:
            return "Regular Engagement"

    segments["Strategy"] = segments["Segment"].apply(strategy)
    return segments