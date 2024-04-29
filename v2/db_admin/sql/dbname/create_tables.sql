create table embeddings(
    id bytea primary key,  -- sha256 of text_
    text_ varchar not null,
    embedding vector(1536) not null
);

create table sources(
    source varchar unique not null,
    embedding_id bytea not null,
    constraint fk_embedding_id
        foreign key (embedding_id)
        references embeddings(id)
        on delete cascade
);
