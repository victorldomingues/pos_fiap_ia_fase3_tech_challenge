# Relatório técnico — Assistente Médico Virtual Acadêmico

## Resumo

O projeto demonstra uma aplicação informacional para protocolos institucionais, com retrieval augmented generation (RAG), fluxo LangChain explícito, fine-tuning comportamental opcional do Qwen2.5-3B-Instruct por QLoRA, CLI, persistência SQLite de dados inteiramente sintéticos, guardrails, auditoria minimizada e revisão humana.

O sistema **não é dispositivo médico nem está pronto para uso clínico**. Não diagnostica, faz triagem, recomenda tratamentos, gera prescrições ou decide condutas. O modelo não inventa, completa ou aprova decisões clínicas.

## Objetivos acadêmicos

1. Demonstrar ingestão, limpeza, anonimização defensiva, validação, deduplicação e particionamento de dados.
2. Explicar embeddings, chunking, ChromaDB e recuperação citável de protocolos.
3. Demonstrar orquestração LangChain LCEL com recusa/fallback e validação de entrada/saída.
4. Apresentar QLoRA/PEFT/BitsAndBytes sem tratar pesos como fonte de conhecimento clínico.
5. Demonstrar perfis, autorização por credencial e isolamento administrativo com dados sintéticos.
6. Mostrar auditoria minimizada, LangSmith com conteúdo omitido e decisões de revisão humana.

## Decisões técnicas

### Conhecimento e geração

- Protocolos versionados são a fonte informacional; cada resposta de protocolo deve citar somente documentos efetivamente recuperados.
- ChromaDB persiste o índice em disco e usa manifesto para validar modelo de embeddings e alterações dos documentos. Os documentos originais continuam independentes do índice.
- Casos/anotações sintéticas ficam separados de protocolos. A recuperação vetorial dessas anotações só ocorre depois de o SQLite autorizar paciente e profissional; o índice é temporário e em memória.
- Qwen2.5-3B-Instruct + QLoRA é opcional e ensina formato, tom e recusas. Os adaptadores são salvos em local persistente e validados antes do uso.
- PubMedQA/MedQuAD são opcionais, sujeitos a termos/licenças e processados para RAG acadêmico não curado em índice separado. Não são protocolos institucionais ou conteúdo clinicamente validado. Respostas médicas desses conjuntos não são incluídas nos splits de fine-tuning.

### Persistência e acesso

- SQLite contém `patients`, `doctors`, `appointments`, `prescriptions`, `clinical_notes` e `audit_logs`; chaves estrangeiras, índices e restrições protegem integridade básica.
- Tokens aleatórios são entregues uma vez; somente SHA-256 é armazenado. A identidade e o papel são resolvidos pelo serviço em cada operação.
- Pacientes acessam apenas seus dados/agendamentos. Profissionais consultam pacientes com vínculo explícito de agenda. Saídas mascaram identificadores e contatos.
- Prescrições e anotações são sempre de autoria humana, com paciente sintético; propostas exigem revisão por outro profissional e não são expostas ao perfil paciente.
- A auditoria funcional registra ação, usuário sintético, perfil, instante e resultado; não armazena texto de pergunta, resposta, anotação ou prescrição.

### Privacidade e observabilidade

- Guardrails são determinísticos e conservadores, mas não garantem detecção de toda PII ou intenção clínica.
- LangSmith é opt-in. Mesmo habilitado, o cliente omite inputs e outputs; apenas estruturas e metadados são enviados. Não há opção que habilite conteúdo clínico.
- Tokens e chaves são configurados pelo ambiente/Secrets. O tracing não deve ser habilitado com dados reais.

## Fluxos e artefatos

O detalhamento Mermaid dos fluxos do assistente, médico, paciente, RAG, fine-tuning, LangChain e HITL está em [arquitetura.md](./arquitetura.md). O [notebook único na raiz](../../IADT%203%20-%20Victor%20Luiz%20Domingues.ipynb) cobre toda a apresentação integrada, inclusive QLoRA no Colab.

| Fase | Artefatos principais |
|---|---|
| Arquitetura | `docs/arquitetura.md`, este relatório |
| SQLite e perfis | `src/database/`, `data/sqlite/`, `scripts/clinic_demo.py` |
| Dados | `scripts/download_public_datasets.py`, `scripts/prepare_data.py`, `src/finetuning/dataset.py` |
| Fine-tuning | `src/finetuning/train_qlora.py`, notebook único na raiz (Parte 2) |
| RAG e ferramentas | `src/rag/`, `src/assistant/clinic_tools.py` |
| Segurança e auditoria | `src/security/`, `src/logging/`, `src/utils/observability.py` |
| QA | `tests/`, `scripts/evaluate.py` |

## Execução local

Dentro de `medical-ai-assistant/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/build_index.py
python -m pytest
```

Para provisionar um médico fictício, configure um segredo temporário fora do repositório e use a mesma configuração nas execuções seguintes:

```bash
export MED_DB_BOOTSTRAP_TOKEN='segredo-local'
python scripts/clinic_demo.py --provision-doctor
python scripts/clinic_demo.py
```

O cadastro de paciente requer confirmar que os dados são sintéticos. Tokens de acesso são credenciais bearer: guarde-os fora do Git e de capturas/logs.

O treinamento QLoRA não faz parte da instalação básica. Abra o notebook único no Colab, ajuste o caminho do projeto no Drive, selecione GPU e habilite o treinamento apenas mediante ação explícita na Parte 2. O adaptador é persistido em `artifacts/qwen-qlora-adapter/`; `/content` não é persistente.

## Verificação e critérios

O conjunto existente de avaliação cobre respostas citadas, bloqueios, ausência de fonte, saída segura e HITL. Os testes novos cobrem cadastro e autenticação, isolamento entre paciente/profissional, mascaramento, conflitos de agenda, autoria humana, revisão por outro profissional e autorização antes de embeddings pessoais.

As métricas não substituem validação clínica. A similaridade vetorial é apenas uma medida de recuperação. Uma resposta ou proposta aprovada por um humano nesta demonstração não implica avaliação clínica independente.

## Limitações e riscos residuais

- O sistema não integra identidade institucional, MFA, recuperação/expiração de credencial, RBAC gerenciado ou controle multi-tenant corporativo.
- Um token roubado permite impersonação local até ser removido do banco; esse protótipo não implementa revogação/expiração.
- Máscaras por regex podem deixar PII residual. Evite inteiramente dados reais de pacientes.
- SQLite e Chroma no Google Drive não oferecem garantias de concorrência, integridade ou backup. Execute um processo e mantenha restauração testada.
- Dados públicos podem ter termos restritivos e fatos médicos fora de contexto; inspecione licença e conteúdo antes de qualquer curadoria ou uso em RAG.
- Modelo generativo pode alucinar ou interpretar mal o contexto. Guardrails e HITL reduzem alguns riscos, não os eliminam.
- A configuração LangSmith prioriza privacidade e, portanto, não registra corpos de prompts/respostas; a cobertura de conteúdo é intencionalmente limitada.
- Não há alegação de conformidade regulatória, segurança clínica certificada ou prontidão para produção.
