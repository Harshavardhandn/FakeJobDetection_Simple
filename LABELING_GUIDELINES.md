# Naukri manual labelling protocol

Labels: `0 = genuine`, `1 = fraudulent`.

Suggested annotation fields: `fraudulent`, `reviewer_id`, `review_date`, `evidence_url`, `evidence_notes`, `confidence`.

1. Select and document a sample of Naukri listings across categories and dates.
2. Verify employer and vacancy using official company career pages and independently sourced contact details.
3. Treat requests for registration fees, deposits or credential disclosure as warning signs, not single-keyword proof.
4. Exclude ambiguous/unverifiable cases from metrics rather than forcing a binary label.
5. Have a second reviewer independently assess a subset and resolve disagreements.
6. Never use model predictions as ground-truth labels (that creates circular evaluation).
