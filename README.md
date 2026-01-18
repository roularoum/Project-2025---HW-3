Εργασία 3 — Αναζήτηση Απομακρυσμένων Ομόλογων με ESM-2 + ANN + BLAST

Ο φάκελος περιέχει πλήρη υλοποίηση του pipeline της Εργασίας 3:

Βήμα 1: Παραγωγή protein embeddings με ESM-2
Μοντέλο: facebook/esm2_t6_8M_UR50D, layer 6, mean pooling πάνω στα residues.

Βήμα 2: Approximate Nearest Neighbor (ANN) αναζήτηση στον χώρο των embeddings με:

Euclidean LSH

Hypercube

IVF-Flat

IVF-PQ

Neural LSH

Βήμα 3: Σύγκριση των αποτελεσμάτων ANN με BLAST (στην ίδια βάση), υπολογίζοντας:

Recall@N

QPS (Queries Per Second)

Παραπομπές προδιαγραφών: assignment3/project.pdf, assignment3/reference.pdf

Περιεχόμενα
Scripts

protein_embed.py: δημιουργεί embeddings για τη βάση (vectors.dat + ids.txt).

protein_search.py: τρέχει ANN μεθόδους + BLAST baseline και παράγει το results.txt στη ζητούμενη μορφή.

protein_grid_search.py: grid search υπερ-παραμέτρων → CSV για καμπύλες Recall vs QPS.

Library code

protein_ann/esm2.py: ESM-2 embedding (mean pooling, τελευταία layer).

protein_ann/ann/: υλοποιήσεις ANN indexes (LSH, Hypercube, IVF-Flat, IVF-PQ, Neural LSH).

protein_ann/blast.py: δημιουργία BLAST DB, εκτέλεση BLAST, parsing αποτελεσμάτων.

protein_ann/output_format.py: utilities για σωστή μορφοποίηση εξόδου.

Δεδομένα

datasets/swissprot_50k.fasta: βάση πρωτεϊνών (50k sequences).

datasets/targets.fasta: queries.

datasets/targets.pfam_map.tsv: mapping για βιολογική αξιολόγηση (στο report).

Απαιτήσεις

Python 3.10+

Linux περιβάλλον (όπως απαιτείται)

NCBI BLAST+ εγκατεστημένο και στο PATH
(εκτελέσιμα: makeblastdb, blastp)

Εγκατάσταση Python dependencies:

pip install -r requirements.txt

Βήμα 1 — Δημιουργία embeddings

Εντολή:

python protein_embed.py \
  -i datasets/swissprot_50k.fasta \
  -o protein_vectors.dat \
  -model esm2_t6_8M_UR50D


Παράγει:

protein_vectors.dat: NumPy array (στην ουσία .npy) με shape (N, 320)

ids.txt: τα IDs (1 ανά γραμμή) ευθυγραμμισμένα με τις γραμμές των vectors

Βήμα 2/3 — Benchmark ANN + σύγκριση με BLAST

Τρέχει όλες τις μεθόδους:

python protein_search.py \
  -d protein_vectors.dat \
  -q datasets/targets.fasta \
  -o results.txt \
  -method all \
  --blast-fasta datasets/swissprot_50k.fasta


Παράδειγμα για μεμονωμένη μέθοδο:

python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method lsh --blast-fasta datasets/swissprot_50k.fasta
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method hypercube --blast-fasta datasets/swissprot_50k.fasta
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method ivf --blast-fasta datasets/swissprot_50k.fasta
python protein_search.py -d protein_vectors.dat -q datasets/targets.fasta -o results.txt -method neural --blast-fasta datasets/swissprot_50k.fasta


Σημειώσεις:

Το BLAST database cache κρατιέται στο .blast_db_cache/ (δημιουργείται αυτόματα).

Το results.txt είναι σε plain text 2 επιπέδων:

Summary: χρόνος/ερώτημα, QPS, Recall@N σε σχέση με BLAST Top-N

Top-K neighbors: ID γείτονα, L2 distance, BLAST identity, αν ανήκει στο BLAST Top-N, και βιολογικό σχόλιο

Grid Search (Recall vs QPS καμπύλες)

Παράδειγμα για LSH:

python protein_grid_search.py \
  -d protein_vectors.dat \
  -q datasets/targets.fasta \
  -o grid_lsh.csv \
  --method lsh \
  --recall-n 50 \
  --lsh-k-grid 2,4,6 \
  --lsh-L-grid 5,10 \
  --lsh-w-grid 2.0,4.0,6.0 \
  --blast-fasta datasets/swissprot_50k.fasta


Το CSV χρησιμοποιείται για plot ώστε να δικαιολογηθούν οι υπερ-παράμετροι στο report.

Report

Η αναφορά βρίσκεται στο report.md και ακολουθεί τη δομή του reference.pdf και τις απαιτήσεις του project.txt.

8. Στοιχεία Φοιτητή

Η εργασία εκπονείται από:

Ονοματεπώνυμο: Ρουμάνη Σπυριδούλα / Σωτηρχέλλη Ευμορφία

Αριθμός Μητρώου (ΑΜ): 1115202000175 / 1115202000187

Email: sdi2000175@di.uoa.gr
 / sdi2000187@di.uoa.gr
