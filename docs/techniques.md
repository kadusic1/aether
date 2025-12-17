# Reminder: Techniques to apply in the project

## 1. Prompt Engineering
Check model docs

## 2. Automatic Rule Scoring
Score this output from 1–10 for virality. If <8, rewrite.

## 3. Tokenization & Context Control

### Token Budgeting
- How many tokens your prompts actually consume
- Why long chat templates reduce quality

**Useful tools:**

```python
tokenizer(...)["input_ids"].shape[-1]
```

📌 Real systems actively manage token length.

## 4. Stop Sequences

Prevent rambling or extra sections.

```python
eos_token_id=tokenizer.eos_token_id
```

## 5. Sampling Strategies (Beyond Temperature)

### Currently Using:
- temperature
- top_p
- repetition_penalty

### Learn Next:
- top_k
- Dynamic temperature (higher → lower per pass)
- Different sampling for headline vs body

This shows intentional generation control, not random tuning.

## 6. Memory & Performance Techniques (Contractor-Level)

### Key Concepts:
- KV-cache awareness
- How attention KV caching works
- Why multi-step generation is expensive without it

Hugging Face handles this, but knowing it matters in interviews.

## 7. CPU Fallback & Graceful Degradation

### Learn:
```python
device_map="auto"
```

- Mixed CPU/GPU loading
- Handling OOM without crashing

## 8. Output Validation & Post-Processing

Professionals don't trust raw LLM output.

### Learn:
- Regex validation
- Length checks
- Forbidden word filters
- Deduplication

**Example:**
```python
if len(items) != 6:
    regenerate()
```

## 9. Logging, Metrics & Evaluation (CV Gold)

### Learn to Log:
- Prompt version
- Seed
- Generation params
- Time per generation

This is what differentiates a hobbyist from an engineer.

## 10. Deployment-Thinking (Even if Local)

Even if this runs locally:

- Separate model loading from logic
- Make functions stateless
- Prepare for API wrapping

This shows production mindset.
