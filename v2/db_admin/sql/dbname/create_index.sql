-- set maintenance_work_mem TO '32 GB';
-- set max_parallel_maintenance_workers TO 16; 
create index embeddings_index
    on embeddings
    using ivfflat (embedding vector_ip_ops)
    with (lists = 2000);
