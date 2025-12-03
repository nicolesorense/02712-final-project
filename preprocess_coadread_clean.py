"""
COADREAD Data Preprocessing for NBS²
Processes TCGA Colorectal Cancer data for network-based supervised stratification

Input files (download from cBioPortal):
- data_mutations.txt
- data_cna.txt  
- data_clinical_patient.txt

Output files:
- COADREAD_training_data.txt
- COADREAD_validation_data.txt
- COADREAD_training_labels.txt
- COADREAD_validation_labels.txt
- COADREAD_feature_names.txt
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# ==================================================
# CONFIGURATION
# ==================================================
INPUT_DIR = 'coadread_tcga/' # Coadread tcga data from cBioPortal
OUTPUT_DIR = 'coadread_processed/'      
NETWORK_FILE = 'data/BRCA_edge2features_2.txt' # NBSS-code/data in repo

TRAIN_SPLIT = 0.67  # 67% training, 33% validation
MIN_MUTATIONS = 4   # Minimum mutations per gene to keep

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("COADREAD Data Preprocessing for NBS²")
print("="*60)

# ==================================================
# STEP 1: Parse Mutation Data
# ==================================================
print("\nStep 1: Parsing mutation data...")

df_mut = pd.read_csv(INPUT_DIR + 'data_mutations.txt', sep='\t', comment='#', low_memory=False)
print(f"  ✓ Loaded {len(df_mut)} mutation records")

# Extract patient IDs (first 12 characters of sample barcode)
df_mut['patient_id'] = df_mut['Tumor_Sample_Barcode'].str[:12]

# Filter out silent mutations
silent_types = ['Silent', '3\'UTR', '5\'UTR', 'Intron', 'IGR']
df_mut = df_mut[~df_mut['Variant_Classification'].isin(silent_types)]
print(f"  ✓ After filtering silent mutations: {len(df_mut)} records")

# Create binary mutation matrix (patients × genes)
df_mut_matrix = df_mut.groupby(['patient_id', 'Hugo_Symbol']).size().unstack(fill_value=0)
df_mut_matrix = (df_mut_matrix > 0).astype(int)
print(f"  ✓ Mutation matrix: {df_mut_matrix.shape[0]} patients × {df_mut_matrix.shape[1]} genes")

# ==================================================
# STEP 2: Parse Copy Number Alterations
# ==================================================
print("\nStep 2: Parsing copy number alterations...")

df_cna = pd.read_csv(INPUT_DIR + 'data_cna.txt', sep='\t', index_col=0)
print(f"  ✓ Loaded CNA data: {df_cna.shape[0]} genes x {df_cna.shape[1]} samples")

# Convert sample IDs to patient IDs
df_cna.columns = df_cna.columns.str[:12]

# Binary encoding: 1 if high-level amplification (+2) or deep deletion (-2)
df_cna_binary = ((df_cna == 2) | (df_cna == -2)).astype(int)
df_cna_binary = df_cna_binary.T  # Transpose to patients x genes
print(f"  ✓ CNA matrix: {df_cna_binary.shape[0]} patients x {df_cna_binary.shape[1]} genes")

# ==================================================
# STEP 3: Combine Mutations and CNAs
# ==================================================
print("\nStep 3: Combining mutation and CNA data...")

# Remove duplicate gene names (keep first occurrence)
if df_cna_binary.columns.duplicated().any():
    n_duplicates = df_cna_binary.columns.duplicated().sum()
    print(f"  ! Removing {n_duplicates} duplicate genes from CNA data")
    df_cna_binary = df_cna_binary.loc[:, ~df_cna_binary.columns.duplicated(keep='first')]

# Get all unique patients and genes
all_genes = sorted(set(df_mut_matrix.columns) | set(df_cna_binary.columns))
all_patients = sorted(set(df_mut_matrix.index) | set(df_cna_binary.index))

print(f"  ✓ Combining {len(all_patients)} patients and {len(all_genes)} genes...")

# Align dataframes and combine with OR operation
df_mut_aligned = df_mut_matrix.reindex(index=all_patients, columns=all_genes, fill_value=0)
df_cna_aligned = df_cna_binary.reindex(index=all_patients, columns=all_genes, fill_value=0)
df_combined = ((df_mut_aligned > 0) | (df_cna_aligned > 0)).astype(int)

print(f"  ✓ Combined matrix: {df_combined.shape[0]} patients x {df_combined.shape[1]} genes")
print(f"  ✓ Mutations: {(df_mut_aligned > 0).sum().sum()} alterations")
print(f"  ✓ CNAs: {(df_cna_aligned > 0).sum().sum()} alterations")

# ==================================================
# STEP 4: Filter to Genes in Network
# ==================================================
print("\nStep 4: Filtering to genes in Pathway Commons network...")

# Load network file to get valid genes
edges = pd.read_csv(NETWORK_FILE, sep='\t', header=None)
network_genes = set(edges[0]).union(set(edges[1]))
print(f"  ✓ Genes in network: {len(network_genes)}")

# Keep only genes that exist in the network
genes_in_both = list(set(df_combined.columns).intersection(network_genes))
df_combined = df_combined[genes_in_both]

print(f"  ✓ Genes after network filtering: {len(genes_in_both)}")
print(f"  ✓ Matrix: {df_combined.shape[0]} patients x {df_combined.shape[1]} genes")

# ==================================================
# STEP 5: Parse Subtype Labels
# ==================================================
print("\nStep 5: Parsing subtype labels...")

df_clinical = pd.read_csv(INPUT_DIR + 'data_clinical_patient.txt', sep='\t', comment='#')
print(f"  ✓ Loaded clinical data for {len(df_clinical)} patients")

# Define histological subtypes based on Cancer Type Detailed
if 'HISTOLOGICAL_DIAGNOSIS' in df_clinical.columns:
    
    # Clean up values to match the 3 main categories
    def clean_histology(hist):
        if pd.isna(hist):
            return None
        hist_str = str(hist)
        if 'Mucinous' in hist_str:
            return 'Mucinous'
        elif 'Rectal' in hist_str:
            return 'Rectal'
        elif 'Colon' in hist_str:
            return 'Colon'
        else:
            return None
    
    df_clinical['SUBTYPE'] = df_clinical['HISTOLOGICAL_DIAGNOSIS'].apply(clean_histology)
    
    # Show distribution
    subtype_counts = df_clinical['SUBTYPE'].value_counts()
    print(f"  Using Cancer Type Detailed (histological subtype)")
    print(f"  ✓ Subtype distribution:")
    for subtype, count in subtype_counts.items():
        if pd.notna(subtype):
            print(f"     {subtype}: {count} patients")
    
    # Keep all 3 histological groups
    df_clinical = df_clinical[df_clinical['SUBTYPE'].notna()].copy()
    print(f"  ✓ Using all 3 histological groups")
    
else:
    raise ValueError("HISTOLOGICAL_DIAGNOSIS column not found in clinical data")

# Create patient-to-subtype mapping
pat_id_col = 'PATIENT_ID' if 'PATIENT_ID' in df_clinical.columns else df_clinical.columns[1]
pat2subtype = dict(zip(df_clinical[pat_id_col], df_clinical['SUBTYPE']))

# Remove patients with missing subtypes
pat2subtype = {k: v for k, v in pat2subtype.items() 
               if pd.notna(v) and str(v) not in ['NA', 'nan', '[Not Available]', '[Unknown]', 'None']}

print(f"  ✓ Assigned subtypes to {len(pat2subtype)} patients")

# Show final distribution
subtype_counts = pd.Series(pat2subtype).value_counts()
print("\n  Final subtype distribution:")
for subtype, count in subtype_counts.items():
    print(f"    {subtype}: {count} patients ({count/len(pat2subtype)*100:.1f}%)")

# ==================================================
# STEP 6: Filter Patients and Genes
# ==================================================
print("\nStep 6: Filtering patients and genes...")

# Keep only patients with both mutation data and subtype labels
patients_with_data = set(df_combined.index)
patients_with_labels = set(pat2subtype.keys())
patients_final = patients_with_data & patients_with_labels

print(f"  ✓ Patients with mutation data: {len(patients_with_data)}")
print(f"  ✓ Patients with subtype labels: {len(patients_with_labels)}")
print(f"  ✓ Patients with both: {len(patients_final)}")

df_combined = df_combined.loc[sorted(patients_final), :]

# Filter genes: keep only genes mutated in ≥4 patients
gene_counts = (df_combined > 0).sum(axis=0)
genes_to_keep = gene_counts[gene_counts >= MIN_MUTATIONS].index
df_combined = df_combined.loc[:, genes_to_keep]

print(f"  ✓ After filtering (≥{MIN_MUTATIONS} mutations/gene): {df_combined.shape[1]} genes")
print(f"  ✓ Final matrix: {df_combined.shape[0]} patients × {df_combined.shape[1]} genes")

# ==================================================
# STEP 7: Train/Validation Split
# ==================================================
print("\nStep 7: Splitting into training/validation sets...")

patients = list(df_combined.index)
labels = [pat2subtype[p] for p in patients]

# Stratified split to maintain subtype proportions
train_patients, val_patients, train_labels, val_labels = train_test_split(
    patients, labels, train_size=TRAIN_SPLIT, stratify=labels, random_state=42
)

print(f"  ✓ Training: {len(train_patients)} patients")
print(f"  ✓ Validation: {len(val_patients)} patients")

# Show distribution in each split
print("\n  Training subtypes:")
for subtype, count in pd.Series(train_labels).value_counts().items():
    print(f"    {subtype}: {count}")

print("\n  Validation subtypes:")
for subtype, count in pd.Series(val_labels).value_counts().items():
    print(f"    {subtype}: {count}")

# ==================================================
# STEP 8: Save Output Files
# ==================================================
print("\nStep 8: Saving output files...")

# Training data
df_train = df_combined.loc[train_patients, :]
df_train.to_csv(OUTPUT_DIR + 'COADREAD_training_data.txt', sep='\t')
print(f"  ✓ {OUTPUT_DIR}COADREAD_training_data.txt")

# Validation data
df_val = df_combined.loc[val_patients, :]
df_val.to_csv(OUTPUT_DIR + 'COADREAD_validation_data.txt', sep='\t')
print(f"  ✓ {OUTPUT_DIR}COADREAD_validation_data.txt")

# Training labels
with open(OUTPUT_DIR + 'COADREAD_training_labels.txt', 'w') as f:
    for label in train_labels:
        f.write(f"{label}\n")
print(f"  ✓ {OUTPUT_DIR}COADREAD_training_labels.txt")

# Validation labels
with open(OUTPUT_DIR + 'COADREAD_validation_labels.txt', 'w') as f:
    for label in val_labels:
        f.write(f"{label}\n")
print(f"  ✓ {OUTPUT_DIR}COADREAD_validation_labels.txt")

# Gene names
with open(OUTPUT_DIR + 'COADREAD_feature_names.txt', 'w') as f:
    for gene in df_combined.columns:
        f.write(f"{gene}\n")
print(f"  ✓ {OUTPUT_DIR}COADREAD_feature_names.txt")

# ==================================================
# SUMMARY
# ==================================================
print("\n" + "="*60)
print("PREPROCESSING COMPLETE!")
print("="*60)
print(f"\nFinal dataset:")
print(f"  - Total patients: {df_combined.shape[0]}")
print(f"  - Total genes: {df_combined.shape[1]}")
print(f"  - Training samples: {len(train_patients)}")
print(f"  - Validation samples: {len(val_patients)}")
print(f"  - Number of subtypes: {len(set(labels))}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nNext step:")
print("  Copy network file: cp <BRCA_edge2features_2.txt> coadread_processed/COADREAD_edge2features.txt")
print("="*60)

# Check which genes are in your data vs the network
import pandas as pd

# Your genes with mutations
mutation_genes = set(df_combined.columns)  # From preprocessing

# Network genes
network_genes

# Overlap
overlap = mutation_genes & network_genes
print(f"Genes in both: {len(overlap)} / {len(mutation_genes)}")
print(f"Coverage: {len(overlap)/len(mutation_genes)*100:.1f}%")

# Are key CRC genes in the network?
crc_key_genes = ['APC', 'KRAS', 'TP53', 'BRAF', 'PIK3CA', 'SMAD4', 'FBXW7']
for gene in crc_key_genes:
    in_network = gene in network_genes
    has_mutations = gene in mutation_genes
    print(f"{gene}: Network={in_network}, Mutations={has_mutations}")

# Are the right genes in the network?
crc_pathways = {
    'WNT': ['APC', 'CTNNB1', 'TCF7L2', 'AXIN2'],
    'RAS': ['KRAS', 'NRAS', 'BRAF'],
    'PI3K': ['PIK3CA', 'PTEN', 'AKT1'],
    'TP53': ['TP53', 'MDM2', 'CDKN2A'],
    'TGF-β': ['SMAD4', 'SMAD2', 'TGFBR2']
}

for pathway, genes in crc_pathways.items():
    in_network = sum(1 for g in genes if g in network_genes)
    has_muts = sum(1 for g in genes if g in mutation_genes)
    print(f"{pathway}: {in_network}/{len(genes)} in network, {has_muts}/{len(genes)} have mutations")
