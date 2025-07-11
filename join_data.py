import polars as pl
from transformers import AutoTokenizer
import json
import re
import gc
import torch
import os
                        
notevents = pl.read_csv_batched("./NOTEEVENTS_REPORTS.csv", has_header=True)
procedures = pl.scan_csv("./PROCEDURES_ICD.csv", has_header=True, dtypes={"ICD9_CODE": pl.String})
diagnoses = pl.scan_csv("./DIAGNOSES_ICD.csv", has_header=True, dtypes={"ICD9_CODE": pl.String})
tokenizer = AutoTokenizer.from_pretrained("samrawal/medical-sentence-tokenizer")

counter = 1
batches = notevents.next_batches(25)

diagnosis_vocab = {}
procedure_vocab = {}

def get_diagnosis(code: str):
    if code not in diagnosis_vocab:
        diagnosis_vocab[code] = len(diagnosis_vocab)
    return diagnosis_vocab[code]

def get_procedure(code: str):
    if code not in procedure_vocab:
        procedure_vocab[code] = len(procedure_vocab)
    return procedure_vocab[code]

def get_SVD(input_list: list, n_components=64):
  input_list = torch.tensor(input_list, dtype=torch.float64)
  # # Create an SVD instance
  # svd = TruncatedSVD(n_components=n_components)

  # # Fit and transform the data
  # reduced_tensor = svd.fit_transform(input_list)

  # # Convert back to tensor if needed
  # reduced_tensor = torch.tensor(reduced_tensor, dtype=torch.float64)
  reduced_tensor = input_list.mean(dim=0).long().tolist()

  del input_list
  #del svd
  gc.collect()

  return reduced_tensor

print(tokenizer.pad_token_type_id)

# with open("./NOTEEVENTS_TOK.csv", "a+", encoding="utf-8") as f:
#   while batches:
#       chunk = pl.concat(batches).select(["SUBJECT_ID", "TEXT"])
#       chunk = chunk.lazy()
#       # Apply transformations using lazy evaluation where possible
#       chunk = chunk.with_columns(pl.col("TEXT").str.replace_all(r"\s+", " ", literal=True)) # Use literal=True for regex with special chars
#       chunk = chunk.with_columns(pl.col("TEXT").map_elements(lambda x: tokenizer(x, return_tensors="pt", truncation=True, padding='max_length')["input_ids"].squeeze().tolist(), return_dtype=pl.List(pl.Int64))) #Explicitly set the return dtype

#       # Explicitly collect with GPU engine only where needed, potentially keeping other steps lazy
#       #chunk = chunk.collect(engine="gpu")

#       notevents_procedures = chunk.join(
#           procedures.select(["SUBJECT_ID", "ICD9_CODE"]), on="SUBJECT_ID", how="left"
#       ).group_by("SUBJECT_ID").agg(
#         pl.col("ICD9_CODE"),
#         pl.col("TEXT")
#       ).with_columns([
#           pl.col("ICD9_CODE").map_elements(lambda x: list(set(x))).alias("ICD9_CODE")
#       ]).collect()

#       #notevents_procedures = chunk.join(notevents_procedures, on="SUBJECT_ID", how="left").collect(engine="gpu")

#       notevents_procedures = notevents_procedures.with_columns(
#           pl.col("TEXT").map_elements(lambda x: get_SVD(x), return_dtype=pl.List(pl.Int64))
#       )

#       notevents_diagnoses = chunk.join(
#           diagnoses.select(["SUBJECT_ID", "ICD9_CODE"]), on="SUBJECT_ID", how="left"
#       ).group_by("SUBJECT_ID").agg(
#           pl.col("ICD9_CODE")
#       ).with_columns(
#           pl.col("ICD9_CODE").map_elements(lambda x: list(set(x))).alias("ICD9_CODE")
#       ).collect()

#       del chunk
#       gc.collect()

#       final = notevents_procedures.join(notevents_diagnoses, on="SUBJECT_ID", how="left")

#       del notevents_procedures
#       del notevents_diagnoses
#       gc.collect()
      
#       final = final.with_columns([
#           pl.col("TEXT").map_elements(lambda x: f"{x.to_list()}", return_dtype=pl.String).alias("TEXT"),
#           pl.col("ICD9_CODE").map_elements(lambda x: f"{[get_diagnosis(i) for i in x if i]}", return_dtype=pl.String).alias("ICD9_PROC"),
#           pl.col("ICD9_CODE_right").map_elements(lambda x: f"{[get_procedure(i) for i in x if i]}", return_dtype=pl.String).alias("ICD9_DIAG")
#       ]).drop("ICD9_CODE", "ICD9_CODE_right")
      
#       final.write_csv(f, include_header=(counter == 1))

#       del final
#       gc.collect()

#       print(f"Batch set {counter} completed")
#       counter += 1
#       batches = notevents.next_batches(10)
#       try:
#         if os.path.getsize(f.name) > (4294967296/2):
#           break
#       except:
#         pass

# with open("diagnosis_vocab.json", "w+") as f:
#     json.dump(diagnosis_vocab, f)

# with open("procedure_vocab.json", "w+") as f:
#     json.dump(procedure_vocab, f)

# ------------------------------------------------------------------------------------

# final_w_dicdproc = notevents_procedures_diagnoses.join(d_procedures.select(["icd9_code", "long_title"]), on="icd9_code", how="left")

# print(final_w_dicdproc.shape)
# #print(final_w_dicdproc.head())

# final = final_w_dicdproc.join(d_diagnoses.select(["icd9_code", "long_title"]).rename({"icd9_code": "icd9_code_diagnoses"}), on="icd9_code_diagnoses", how="left", suffix="_diagnoses")

# print(final.shape)
# print(final.head())

# final = final.select(["subject_id", "text", "icd9_code", "long_title", "icd9_code_diagnoses", "long_title_diagnoses"]).rename({"icd9_code": "icd9_proc", "icd9_code_diagnoses": "icd9_diag", "long_title": "proc_title", "long_title_diagnoses": "diag_title"})
# final = final.filter((pl.col("icd9_proc").is_not_null() & pl.col("proc_title").is_not_null()) & (pl.col("icd9_diag").is_not_null() & pl.col("diag_title").is_not_null()))
# print(final.shape)
# final.write_csv("./FULL_DATA.csv")
