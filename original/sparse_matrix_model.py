# Usage: python storage_model.py

import numpy as np
import time
from scipy.sparse import dok_matrix, csr_matrix, csc_matrix, coo_matrix

# 1. Generate sample data
n_points = 1_000_000
max_node_id = 100_000
max_distance = 1_000_000

np.random.seed(42)
src = np.random.randint(0, max_node_id, n_points, dtype=np.int32)
dst = np.random.randint(0, max_node_id, n_points, dtype=np.int32)
dist = np.random.randint(0, max_distance, n_points, dtype=np.int32)

# 2. Create python dictionary as model
data_dict = {(int(s), int(d)): int(t) for s, d, t in zip(src, dst, dist)}

# 3. Query latency for dictionary
dict_total_time = 0
dict_query_latency = 0
loop_start_time = time.perf_counter()
for s, d in zip(src, dst):
    start_time = time.perf_counter()
    _ = data_dict[(int(s), int(d))]
    dict_query_latency += time.perf_counter() - start_time

dict_total_time = time.perf_counter() - loop_start_time
print(f"Dict total time: {dict_total_time:.4f}s, query latency: {1000_000*dict_query_latency/n_points:.8f}us, effective latency: {1000_000*dict_total_time/n_points:.8f}us")

# 4. Sparse matrix (DOK)
dok = dok_matrix((max_node_id+1, max_node_id+1), dtype=np.int32)
for s, d, t in zip(src, dst, dist):
    dok[s, d] = t

dok_total_time = 0
dok_query_latency = 0
loop_start_time = time.perf_counter()
for s, d in zip(src, dst):
    start_time = time.perf_counter()
    _ = dok[s, d]
    dok_query_latency += time.perf_counter() - start_time

dok_total_time = time.perf_counter() - loop_start_time
print(f"DOK total time: {dok_total_time:.4f}s, query latency: {1000_000*dok_query_latency/n_points:.8f}us, effective latency: {1000_000*dok_total_time/n_points:.8f}us")

# 5. Sparse matrix (CSR)
csr = dok.tocsr()
csr_total_time = 0
start_time = time.perf_counter()
_ = csr[src, dst].A1
csr_total_time = time.perf_counter() - start_time
# Compute the space used by the CSR matrix in MB
csr_nbytes = csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes
print(f"CSR matrix size: {csr_nbytes / (1024 * 1024):.2f} MB")
print(f"CSR total time: {csr_total_time:.4f}s, query latency: {1000_000*csr_total_time/n_points:.8f}us, size={csr_nbytes/(1024*1024):.2f}MB")

# 6. Sparse matrix (CSC)
csc = dok.tocsc()
start_time = time.perf_counter()
_ = csc[src, dst].A1
csc_total_time = time.perf_counter() - start_time
# Compute the space used by the CSC matrix in MB
csc_nbytes = csc.data.nbytes + csc.indptr.nbytes + csc.indices.nbytes
print(f"CSC matrix size: {csc_nbytes / (1024 * 1024):.2f} MB")
print(f"CSC total time: {csc_total_time:.4f}s, query latency: {1000_000*csc_total_time/n_points:.8f}us, size={csc_nbytes/(1024*1024):.2f}MB")

# 7. Sparse matrix (COO)
coo = dok.tocoo()
# COO does not support efficient random access, so we convert to CSR for batch lookup
coo_csr = coo.tocsr()
start_time = time.perf_counter()
_ = coo_csr[src, dst].A1
coo_total_time = time.perf_counter() - start_time
# Compute the space used by the COO matrix in MB
coo_nbytes = coo.data.nbytes + coo.row.nbytes + coo.col.nbytes
print(f"COO matrix size: {coo_nbytes / (1024 * 1024):.2f} MB")
print(f"COO (via CSR) total time: {coo_total_time:.4f}s, query latency: {1000_000*coo_total_time/n_points:.8f}us, size={coo_nbytes/(1024*1024):.2f}MB")

## OUTPUT:
# Dict total time: 1.0693s, query latency: 0.69634433us, effective latency: 1.06928832us
# DOK total time: 10.1452s, query latency: 9.67363622us, effective latency: 10.14517668us
# CSR matrix size: 8.01 MB
# CSR total time: 0.0615s, query latency: 0.06145535us, size=8.01MB
# CSC matrix size: 8.01 MB
# CSC total time: 0.0588s, query latency: 0.05881621us, size=8.01MB
# COO matrix size: 11.44 MB
# COO (via CSR) total time: 0.0592s, query latency: 0.05924252us, size=11.44MB
