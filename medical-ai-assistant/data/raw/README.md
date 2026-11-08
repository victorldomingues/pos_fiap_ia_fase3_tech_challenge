# data/raw

Coloque aqui os datasets públicos (não versionados):

- `pubmedqa.jsonl` — PubMedQA (`pqa_labeled`): campos `pubid`, `question`, `context.contexts`, `long_answer`.
- `medquad.jsonl` (ou `.csv`) — MedQuAD: campos `question`, `answer` (opcional `document_source`).

Para baixar amostras: `python scripts/download_public_datasets.py --limit 500` (requer `pip install datasets`).
Confira licença, termos de uso e conteúdo antes de baixar ou redistribuir. O download não valida
anonimização, não certifica segurança clínica e não transforma os dados em fonte institucional.
Use `python scripts/prepare_data.py --pubmedqa data/raw/pubmedqa.jsonl --medquad data/raw/medquad.jsonl`
para gerar `data/processed/public_sources.jsonl`. O corpus pode ser indexado separadamente para RAG
acadêmico não curado com `python scripts/build_index.py --with-public-sources`; execute a aplicação
com `MED_PUBLIC_RAG_ENABLED=true`. O campo `approved_for_rag: false` indica que não houve aprovação:
não transforma os registros em fonte institucional ou evidência clínica validada. As respostas médicas
não entram nos splits de fine-tuning. Filtros heurísticos são incompletos; todo conteúdo requer validação
humana e não serve para uso clínico. O corpus pode conter texto clínico de terceiros e permanece ignorado
pelo Git. Nunca coloque prontuários ou dados reais de pacientes aqui.
