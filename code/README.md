# Code - Facebook Fake News

## Directory content

- **Scraper.py**: retrieves Facebook posts. Feature extraction can use the supplied JSON files, so scraping is optional.
- **PostVectorized.py**: defines the feature extraction functions and the `postVectorized` container. It loads spaCy's English model when imported.
- **PostVectorization.py**: reads posts from the `FB` PostgreSQL database and inserts feature arrays into `FB_vectors`.

## Feature-vector structure

Each extraction function expects `(post_id, post_data, label)`, where `post_data` is a dictionary containing the scraped post. The functions read `post[1]`; the database script also uses `post[0]` as the ID and `post[2]` as the label. The six arrays contain **36 features** in this order:

| Feature set | Function | Size | Values, in order |
| --- | --- | --- | --- |
| Reactions | `vectorizingReactions` | 6 | `like`, `haha`, `angry`, `sad`, `love`, `wow` counts |
| Sentiment | `vectorizingVaderSentiment` | 4 | VADER negative, neutral, positive, compound scores |
| Basic information | `vectorizingBasicInfo` | 5 | Shares, comments, total reactions, number of images, number of links |
| Part of speech (POS) | `vectorizingPOS` | 7 | `NOUN`, `PROPN`, `VERB`, `NUM`, `ADJ`, `CONJ`, `ADV` token counts |
| Named entities (NER) | `vectorizingNER` | 7 | `PERSON`, `DATE`, `ORG`, `MONEY`, `GPE`, `NORP`, `EVENT` entity counts |
| Text lengths | `vectorizingTextLengths` | 7 | Tokens, token characters, uppercase tokens, uppercase characters, sentences, characters per token, tokens per sentence |

Sentiment, POS, NER, and text-length features use `post_data["text"]`. POS counts tokens; NER counts entity spans, so a multiword entity contributes one count. Basic information uses `shares`, `comments`, `reaction_count`, `images`, and `links`. Missing reaction types are assigned zero.

The text-length function calls its token count `words`, but spaCy tokens also include punctuation. Its character count is the sum of token lengths, rather than the length of the original string. Uppercase tokens are identified with Python's `str.isupper()`. Empty-text averages are set to zero when division is undefined. The extraction code does not scale or normalize features.

## Creating feature vectors from JSON

### 1. Install dependencies

Run these commands from the repository root, with Python and the `venv` module available:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install spacy vaderSentiment
python -m spacy download en_core_web_sm
```

The model download requires internet access. The repository does not pin dependency or model versions, so new installations may produce different NLP counts from the supplied vectors.

### 2. Extract and export the vectors

Run the following from the repository root in the same environment. It uses the existing functions directly and writes CSV files to `outputs/feature_vectors`. It does not require PostgreSQL.

The example assigns `True` to `real_news.json` and `False` to `fake_news.json`. These labels represent the source-page groups described in the [dataset documentation](../dataset/README.md); they are not individual fact-check verdicts.

```bash
python - <<'PY'
import csv
import json
import sys
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path("code").resolve()))
from PostVectorized import (
    vectorizingReactions, vectorizingVaderSentiment, vectorizingBasicInfo,
    vectorizingPOS, vectorizingNER, vectorizingTextLengths,
)

# Keep the original CSV column names and their order.
feature_sets = [
    ("reaction_vector", vectorizingReactions,
     ["like", "haha", "angry", "sad", "love", "wow"]),
    ("sentiment_vector", vectorizingVaderSentiment,
     ["negative", "neutral", "positive", "compound"]),
    ("basicinfo_vector", vectorizingBasicInfo,
     ["shares", "comments", "reactions", "images", "links"]),
    ("pos_vector", vectorizingPOS,
     ["nouns", "pronouns", "verbs", "numbers", "adjectives", "conjunctions", "adverbs"]),
    ("ner_vector", vectorizingNER,
     ["person", "date", "org", "money", "gpe", "norp", "event"]),
    ("textlenghts_vector", vectorizingTextLengths,
     ["words", "chars", "u_words", "u_chars", "sentences", "avg_word_length", "avg_sent_length"]),
]
columns = [column for _, _, names in feature_sets for column in names]
output_dir = Path("outputs/feature_vectors")
output_dir.mkdir(parents=True, exist_ok=True)
written = skipped = 0

with ExitStack() as stack:
    writers = {}
    headers = [(name, names) for name, _, names in feature_sets]
    headers += [("post_all_vectors", columns), ("all_no_reactions", columns[6:])]
    for name, names in headers:
        handle = stack.enter_context(
            (output_dir / f"{name}.csv").open("w", newline="", encoding="utf-8")
        )
        writers[name] = csv.writer(handle)
        writers[name].writerow(names + ["class"])

    for filename, label in [("real_news.json", True), ("fake_news.json", False)]:
        posts = json.loads((Path("dataset/json") / filename).read_text(encoding="utf-8"))
        for post_data in posts:
            post_id = post_data.get("post_id", "unknown")
            post = (post_id, post_data, label)
            try:
                vectors = [extract(post) for _, extract, _ in feature_sets]
            except Exception as error:
                skipped += 1
                print(f"Skipped {filename}, post {post_id}: {error}", file=sys.stderr)
                continue
            for (name, _, _), vector in zip(feature_sets, vectors):
                writers[name].writerow(vector + [label])
            combined = [value for vector in vectors for value in vector]
            writers["post_all_vectors"].writerow(combined + [label])
            writers["all_no_reactions"].writerow(combined[6:] + [label])
            written += 1

print(f"Wrote {written} posts; skipped {skipped}. Output: {output_dir}")
PY
```

Each CSV row represents one post, with `class` as its final column. `post_all_vectors.csv` contains 36 features plus the label. `all_no_reactions.csv` contains 30 features plus the label: it removes the six reaction-type counts but retains the total reaction count in basic information. All eight files share the same row order. IDs are not exported as features.

The example reports extraction failures and omits the affected post from every output file. Inspect these messages before training a model. Rerunning the example overwrites files in `outputs/feature_vectors`.

### Reproduction details

- The original CSV header `pronouns` corresponds to `PROPN` (proper nouns), not pronouns. The example preserves that header for compatibility.
- The POS function counts the literal tag `CONJ`. Models using `CCONJ` or `SCONJ` will not contribute to that count.
- The reaction function reads `haha`, while some supplied JSON records use `ahah`. Those records receive a zero in the `haha` position. The `hug` reaction is not included in the six reaction features.
- The original database schema declares `textLengthsVector` as an integer array even though its final two values are averages. The supplied CSVs contain integer averages; this direct export preserves the floating-point values returned by the function. It therefore does not promise an exact recreation of the committed CSVs.
- The supplied JSON files contain 4,934 posts; the committed CSVs each contain 4,927 rows. `PostVectorization.py` catches exceptions without reporting them, so the source code alone does not establish why those seven posts were omitted.

## Original PostgreSQL workflow

To run `PostVectorization.py`, you also need PostgreSQL and its additional Python imports:

```bash
python -m pip install psycopg2-binary pandas matplotlib
```

1. Create the source database `FB` and destination database `FB_vectors` in PostgreSQL.
2. Load posts into a source table named `post`. The script uses `SELECT *` and positional indexing, so the first three columns must be the post ID, a JSON/JSONB post object decoded as a Python dictionary, and a Boolean label, in that order. A compatible schema is shown below.
3. Update both `psycopg2.connect(...)` calls in `PostVectorization.py` with your database names, user, password, host, and port. The committed values are placeholders.
4. Run `python code/PostVectorization.py` from the repository root. It creates `post_vector` in the destination database and inserts six arrays and a label for each successfully processed post.

```sql
-- Run in the source database FB before loading the JSON records.
CREATE TABLE post (
    post_id TEXT PRIMARY KEY,
    post_data JSONB,
    label BOOLEAN
);
```

The repository does not include a JSON-to-database loader or a database-to-CSV/ARFF export script. Each raw JSON record must be inserted as `post_data`, with its `post_id` and chosen label in the other two columns. Use the direct JSON example above if you only need CSV vectors.

The destination table must not already exist: the original script's `CREATE TABLE` has no `IF NOT EXISTS` clause. It also silently skips extraction or insertion failures, so a `done` message alone does not confirm that every post was saved.
