# Plano de demonstração (banca) — ~10 minutos

**Preparação (antes):** abra o [notebook único](../../IADT%203%20-%20Victor%20Luiz%20Domingues.ipynb) e prepare os dados. A Parte 2 exige GPU para QLoRA e salva adaptadores no Drive; deixe treinamento desligado na apresentação e valide artefatos existentes. Para CLI, execute `pip install -r requirements.txt`, `python scripts/build_index.py`, `python scripts/clinic_demo.py --automated-tests` e `python -m pytest` em `medical-ai-assistant/`. Cada cenário automatizado imprime entradas e saída diretamente no console/output do Colab; não grava transcript em arquivo e não substitui a suíte `pytest`. Use somente registros fictícios; para o fluxo SQLite interativo, configure `MED_DB_BOOTSTRAP_TOKEN` para provisionar o médico. A auditoria registra metadados minimizados, não pergunta ou resposta clínica.

| Min | Etapa | O que mostrar | Mensagem-chave |
|---|---|---|---|
| 0–1 | Problema e princípios | Slide: apoio informacional; sem diagnóstico/prescrição/dose; humano no circuito | Segurança clínica antes de capacidade |
| 1–2 | Arquitetura | Diagramas de `docs/arquitetura.md` | RAG é a fonte de verdade; fine-tuning só para comportamento; só LangChain (LCEL) |
| 2–3 | Dados | Notebook único, Parte 1: anonimização, validação, split | Sintético + anonimização + descarte de exemplos inseguros |
| 3–4 | Fine-tuning | Notebook único, Parte 2: config QLoRA, `training_metadata.json`, `validate_adapter` | Persistido no Drive, sem retreino ao reutilizar |
| 4–6 | **Demo CLI automatizada** | Parte 6 do notebook ou `python scripts/clinic_demo.py --automated-tests` | Seis cenários sem entrada manual, transcript redigido, SQLite temporário |
| 6–7 | Guardrails | Pedidos de dose/diagnóstico/CPF bloqueados | Filtros de entrada e saída |
| 7–8 | Auditoria e LangSmith | `/auditoria`, abrir `logs/audit.jsonl`; trace LangSmith com metadados | Rastreabilidade minimizada |
| 8–9 | Testes e avaliação | `pytest` e `scripts/evaluate.py` | Métricas e limites |
| 9–10 | Limitações e próximos passos | [README → Limitações](../../README.md#limitações) | Honestidade sobre riscos |

## Roteiro da demo CLI

1. `Qual é o horário de visitas?` → resposta com `[DOC-1]`, protocolo PROT-005 v1.1, confiança, "Validação humana obrigatória.", status PENDENTE.
2. `/pendentes` → mostra o ID; `/aprovar <id> conforme protocolo` → status aprovada.
3. `Como higienizar as mãos?` → outro protocolo e trecho.
4. `Qual a dose de paracetamol para adultos?` → bloqueio (dose), nenhuma chamada ao LLM.
5. `Será que tenho diabetes?` → bloqueio (diagnóstico).
6. `Meu CPF é 111.444.777-35` (sintético) → bloqueio (PII); mostrar no log que o CPF foi redigido.
7. `Qual o protocolo de transplante de córnea?` → "sem fonte confiável".
8. `/auditoria` → "Integridade dos logs: OK". Opcional: editar uma linha do log e repetir para mostrar a detecção.

## Plano B

- Sem GPU/Internet: `MED_LLM_BACKEND=extractive` (padrão) funciona offline após o download dos embeddings.
- Falha do Colab: usar a saída previamente salva do notebook único (Parte 2), se disponível.
- Sem LangSmith: mostrar `build_callbacks` retornando `[]` (desligado por padrão) e explicar a decisão de privacidade.
