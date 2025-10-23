# Reflection

1. **Preprocessing:**
   After applying normalization and basic noise reduction, the transcription accuracy improved a lot. Before preprocessing, the confidence score was around **0.73**, but after, it went up to **0.88**. The background noise in my audio made some words unclear, and normalization helped balance the volume. This made the speech recognition system detect words more accurately.

2. **PII Detection:**
   Detecting and redacting personally identifiable information (like my name and credit card number) was a bit tricky. Sometimes, the detector flagged normal words as names (false positives), and once it missed a number when I spoke too fast (false negative). Overall, it worked well after a few adjustments to the regex and thresholds.

3. **Confidence Scoring:**
   Out of the three factors—API confidence score, SNR, and perplexity—the **API confidence** was the most reliable. The combined score showed a strong correlation with it. My highest combined accuracy came from results where the API confidence was above **0.85**, showing that it best reflected the actual quality of transcription.

4. **Production Considerations:**
   If I were to deploy this pipeline in production, I’d focus on **scalability** by adding batch processing and cloud storage. For **security**, I’d encrypt all PII data and limit access through authentication. For **user experience**, I’d add clear progress feedback and allow users to re-record low-quality audio segments automatically.

